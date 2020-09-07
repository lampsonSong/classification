import os
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from src.transforms import *
import numpy as np

class FaceMaskDataset(Dataset):
    def __init__(self, root_dir, hyp, in_size=224, mode='train'):
        super(FaceMaskDataset, self).__init__()
        self.data_dir = os.path.join(root_dir, mode)
        self.mode = mode 
        self.set_name = set
        self.load_categories()
        self.input_size = in_size
        self.one_hot = np.eye(2)
        self.hyp = hyp


    def load_categories(self):
        names = ['face', 'face_mask']
        self.images = []
        for i, name in enumerate(names):
            category = i
            dir = os.path.join(self.data_dir, name)
            self.images.extend([{'image': os.path.join(dir, file), 'category': category} for file in os.listdir(dir)])            
    
    def __len__(self):
        return len(self.images)

    def transform(self, img):
        if self.mode == 'train':
            # horizontal flip
            img = horizontal_flip(img, p=0.5, order='HWC')

            # blur 
            img = img_blur(img)

            # # rotate
            angle = np.random.randint(self.hyp['rotate_degrees'][0], self.hyp['rotate_degrees'][1])
            img = rotate_bound(angle, img)
            
            # resize
            img = random_sized_crop(img, size=self.input_size, area_frac=0.8)
            
            ## hsv
            #img = augment_hsv(img, h_gain = self.hyp['hsv_h'], s_gain = self.hyp['hsv_s'], v_gain= self.hyp['hsv_v'])

        else:
            img = scale(self.input_size, img)
            img = center_crop(self.input_size, img)
            
        # img = scale(self.input_size, img)
        # img = center_crop(self.input_size, img)
        img = bgr_to_gray(img)
        img = img.transpose([2,0,1]) / 255
        return img

    def mixup(self, item, target, gamma):
        perm_input = None
        while perm_input is None:
            perm = np.random.randint(0, self.__len__())
            perm_input = cv2.imread(self.images[perm]['image'])
            category = self.images[perm]['category']
            perm_target = self.one_hot[category]
        perm_input = perm_input.astype(np.float32, copy=False)
        perm_input = self.transform(perm_input)


        mixed_item = item * gamma + perm_input * (1 - gamma)
        mixed_target = target * gamma + perm_target * (1 - gamma)

        return mixed_item, mixed_target

    def __getitem__(self, index):
        img = cv2.imread(self.images[index]['image'])
        category = self.images[index]['category']
        target = self.one_hot[category]
        # target = category

        while img is None:
            index = (index + 1) % len(self.images)
            img = cv2.imread(self.images[index]['image'])
        img = img.astype(np.float32, copy=False)

        img = self.transform(img)
        #if self.mode == 'train':
        #    img ,target = self.mixup(img, target, 0.5)
        
        return img, target

if __name__ == '__main__':
    hyp = {
            'rotate_degrees' : [0,360],
            'hsv_h' : 0.0138,
            'hsv_s' : 0.678,
            'hsv_v' : 0.36,
        }

    dataset = FaceMaskDataset(root_dir='/media/hsw/E/datasets/CropedVOC', hyp=hyp)
    # dataset = FaceMaskDataset(root_dir='/home/lampson/2T_disk/Data/FaceMask/FaceMaskDataset/classification/CropedVOC', hyp=hyp)
    training_params = {"batch_size": 1,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 0}
    training_generator = DataLoader(dataset, **training_params)
    
    error_count = 0
    for i, (images, target) in enumerate(training_generator):
        print (images.shape)
        # print (target)

        # img = images[0,...]
        # img = img.numpy().transpose([1,2,0])
        # cv2.imshow("-", img)
        # cv2.waitKey(0)
        #break
