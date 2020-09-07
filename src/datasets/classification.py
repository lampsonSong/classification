import os
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from src.transforms import *
import numpy as np
import torch

class ClassifyDataset(Dataset):
    def __init__(self, root_dir, category_names, hyp, loss_type='sigmoid', in_size=224, mode='train'):
        
        super(ClassifyDataset, self).__init__()
        
        assert type(category_names) == type([])
        
        self.data_dir = os.path.join(root_dir, mode)
        self.names = category_names
        self.mode = mode 
        self.set_name = set
        self.load_categories()
        self.input_size = in_size
        self.hyp = hyp
        self.target_type = loss_type 
        self.batch_count = 0
        self.scale_size = in_size

    def load_categories(self):
        names = self.names
        self.one_hot = np.eye(len(names))
        self.images = []
        for i, name in enumerate(names):
            category = i
            dir = os.path.join(self.data_dir, name)
            self.images.extend([{'image': os.path.join(dir, file), 'category': category} for file in os.listdir(dir)])            
    
    def __len__(self):
        return len(self.images)

    def transform(self, img):
        if self.mode == 'train':           
            ## rotate
            #angle = np.random.randint(self.hyp['rotate_degrees'][0], self.hyp['rotate_degrees'][1])
            #img = rotate_bound(angle, img)
            
            ## resize
            #img = cv2.resize(img, (224, 448))
            
            # resize
            img = random_sized_crop(img, size=self.scale_size, area_frac=0.8)

            ## contrast and brightness
            #img = contrast_and_brightness(img, alpha=[0.5, 1.2], beta=[-30,30])

            # horizontal flip
            img = horizontal_flip(img, p=0.5, order='HWC')

            # blur 
            img = img_blur(img)

            ## hsv
            #augment_hsv(img, h_gain = self.hyp['hsv_h'], s_gain = self.hyp['hsv_s'], v_gain= self.hyp['hsv_v'])
        else:
            img = scale(self.input_size, img)
            img = center_crop(self.input_size, img)

        img = np.ascontiguousarray(img, dtype=np.float32)
        #img = bgr_to_gray(img)
        img = norm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.transpose([2,0,1])
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

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))

        t_list = [torch.tensor([t]) for t in targets]
        targets = torch.cat(t_list, 0)
        
        #targets = torch.stack([t for t in targets])
        imgs = torch.stack([torch.from_numpy(img) for img in imgs])

        if(self.hyp['multiscale_training'] and self.batch_count % 10 == 0):
            scale_factor = math.ceil(np.random.rand() * 20) - 10
            self.scale_size = self.input_size + scale_factor * 8

        self.batch_count += 1

        return imgs, targets

        

    def __getitem__(self, index):
        img = cv2.imread(self.images[index]['image'])
        category = self.images[index]['category']

        if self.target_type == 'sigmoid':
            target = self.one_hot[category]
        else:
            target = category

        while img is None:
            index = (index + 1) % len(self.images)
            img = cv2.imread(self.images[index]['image'])
        
        img = img.astype(np.float32, copy=False)
        img = self.transform(img)
        
        return img, target
