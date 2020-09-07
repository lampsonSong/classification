import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import collections

from src.networks.mobilenet.mobilenetv3 import MobileNetV3Tiny, MobileNetV3
from src.networks.regnets.regnet import RegNetY
from src.networks.efficientnet_pytorch.model import EfficientNet
from src.transforms import scale, center_crop, bgr_to_gray, norm
from src.networks.darknet.cspDarknet53 import CSPDarkNet53
from center_face.centerface import CenterFace

from skimage import transform as trans
import argparse

img_scale = 3/4

tmp_points = np.array([
      [30.2946+8.0, 51.6963],
      [65.5318+8.0, 51.5014],
      [48.0252+8.0, 71.7366],
      [33.5493+8.0, 92.3655],
      [62.7299+8.0, 92.2041]], dtype=np.float32) * 2

def alignface(img, points):
    tform = trans.SimilarityTransform()
    tform.estimate(points, tmp_points)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (args.size, args.size), borderValue=0.0)
    cv2.imshow("-", warped)
    # print('M: ', M)
	
    return warped

def get_face_classification_model(model_path):
    
    num_classes = 2
    model = MobileNetV3Tiny(n_class=num_classes, version=args.v)
    #model = MobileNetV3(n_class = 3, mode="small")
    #model = CSPDarkNet53(num_classes=3, used_blocks_num=3, divider=8)
    #model = EfficientNet.from_name(model_name='efficientnet-b0')
    # model = RegNetY(24, 36, 2.5, 13, 1, 8, 2, 4)
    # model = MobileNetV3()

    checkpoints = torch.load(model_path)
    new_checkpoints = collections.OrderedDict()
    for k, v in checkpoints.items():
        name = k[7:] # remove `module.`
        if k[:7] == 'module.':
            new_checkpoints[name] = v
        else:
            new_checkpoints = checkpoints
            break

    model.load_state_dict(new_checkpoints, strict=True)
    model.cuda()
    model.eval()

    return model

import time

def get_face_detect_results(img):
    frame = img
    h, w = frame.shape[:2]
    # detector = CenterFace()
    dets, lms = detector(frame, h, w, threshold=0.25)
    
    results = []
    for i, det in enumerate(dets):
        boxes, scores = det[:4], det[4]
        boxes = np.array(boxes, dtype=np.int32)
        
        tmp = {}
        tmp['box'] = boxes
        tmp['joints'] = lms[i]
        tmp['class'] = -1
        tmp['score'] = 0
        results.append(tmp)


    return results

def get_face_classify_results(img, face_det_res):
    ori = img.copy()
    h,w,_ = ori.shape

    for i, single_face in enumerate(face_det_res):
        box = single_face['box']
        joints = single_face['joints']

        # cx, cy = (box[0] + box[2])/2, (box[1]+box[3])/2
        # bw, bh = box[2]-box[0], box[3]-box[1]
        # bw *= 1.5
        # bh *= 1.5
        # box[0] = cx - bw / 2
        # box[1] = cy - bh / 2
        # box[2] = cx + bw / 2
        # box[3] = cy + bh / 2

        # print ('5 points: ', joints)
        joints = np.reshape(np.array(joints), (-1,2))

        box[0] = int(box[0] / img_scale)
        box[2] = int(box[2] / img_scale)
        box[1] = int(box[1] / img_scale)
        box[3] = int(box[3] / img_scale)
        single_face['joints'] = joints / img_scale

        box[0] = max(0, min(w, box[0]))
        box[2] = max(box[0]+1, min(w, box[2]))
        box[1] = max(0, min(h, box[1]))
        box[3] = max(box[1]+1, min(h, box[3]))

        if box[0] + 10 >= box[2]:
            continue
        if box[1] + 10 >= box[3]:
            continue

        box = np.array(box, dtype=np.int32)
        crop = ori[box[1]:box[3]+1, box[0]:box[2]+1].copy()

        # warp = alignface(ori, joints)
        # crop = warp.copy()
        #crop = cv2.imread("./mask_test.bmp")

        crop = cv2.resize(crop, (args.size, args.size))
        #crop = scale(self.input_size, crop)
        #crop = center_crop(self.input_size, crop)
        crop = crop.astype(np.float32)
        #crop = bgr_to_gray(crop)
        crop = norm(crop)
        crop = crop.transpose([2,0,1])
        crop = np.expand_dims(crop, 0)
        crop = torch.from_numpy(crop).cuda()

        with torch.no_grad():
            output = classifier(crop).cpu().numpy()

        # scores = 1 / (1 + np.exp(-output[0]))
        # no_mask_score = 1 / (1 + np.exp(-output[0][0]))
        # mask_score = 1 / (1 + np.exp(-output[0][1]))

        # print (output[0], no_mask_score, mask_score)


        category = output[0].argmax()
        if args.loss == "sigmoid":
            score = 1 / (1 + np.exp(-output[0][category]))
        elif args.loss == "softmax":
            all_exp = sum([np.exp(output[0][i]) for i in range(len(output[0]))])
            score = np.exp(output[0][category]) / all_exp

        # if score < args.score:
        #     txt = 'Score: {}, Category: Not mask.'
        # else:      
        #     txt = 'Score: {}, Category: Mask'
        
        category = category if score > args.score else -1


        face_det_res[i]['class'] = category
        face_det_res[i]['score'] = score
    
    return face_det_res

def inference(det_img, img):
    
    face_det_res = get_face_detect_results(det_img)
    face_det_res = get_face_classify_results(img, face_det_res)
    colors = [[255,0,0], [0,0,255], [0,255,0], [0,255,255]]

    for face in face_det_res:
        box = face['box']
        joints = face['joints']
        category = face['class']
        score = face['score']

        #if category == 0:
        #    print ('No mask, Score: ', score)

        box = np.array(box, dtype=np.int32)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colors[category+1], 2)
        
        joints = np.array(joints, dtype=np.int32).reshape(-1, 2)
        for point in joints:
            cv2.circle(img, (point[0], point[1]), 3, (0,255,0), -1)

        txt = 'C: ' + str(category) + '_' + str(score)[:4]
        cv2.putText(img, txt, (box[0], box[1]-2), cv2.FONT_HERSHEY_COMPLEX, 1, colors[category+1], 1)
    
    return img

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--score', type=float, default=0.5)
parser.add_argument('--video', type=str, default=None)
parser.add_argument('--loss', type=str, default="sigmoid", help="sigmoid | softmax")
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--v', type=int, default=2)

args = parser.parse_args()

classifier = get_face_classification_model(args.model)
detector = CenterFace()

#img_path = '/home/lampson/2T_disk/Data/images/mask_test'
#
#for file in os.listdir(img_path):
#    img = cv2.imread(os.path.join(img_path, file))
#    ori = inference(img)
#    cv2.imshow('reslult', ori)
#    if cv2.waitKey(0) == ord('q'):
#        break


idx = 0
video_path = args.video if args.video else 0
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    det = cap.grab()
    
    #idx += 1
    #if idx % 2 == 0:
    #    pass
    
    if det:
        flag, frame = cap.retrieve()
        start = time.time()
        infer_frame = cv2.resize(frame, ( int(frame.shape[1] * img_scale), int(frame.shape[0] * img_scale) ) )
        #print(infer_frame.shape)
        # frame = cv2.imread('/media/hsw/E/work/ulsee/face_mask_classification/test.jpg')
        #infer_frame = frame
        ori = inference(infer_frame, frame)
        end = time.time()
        print("- duration : ", end-start)
        cv2.imshow('result', ori)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    else:
        print ('No input image.')
        break
