"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import os
import re
from torch.utils.data import Dataset, DataLoader
from src.transforms import *
from src.networks.regnets.config import *


class Imagenet(Dataset):
    def __init__(self, root_dir, mode="train"):
        if mode == 'train':
            self.data_dir = os.path.join(root_dir, 'ILSVRC2012_img_train')
        else:
            self.data_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.set_name = set
        self.load_categories()

    def load_categories(self):
        self.raw_category_ids = sorted(file_ for file_ in os.listdir(self.data_dir) if re.match(r"^n[0-9]+$", file_))
        self.fine_category_ids = {value: key for key, value in enumerate(self.raw_category_ids)}
        self.images = []
        for raw_id in self.raw_category_ids:
            fine_id = self.fine_category_ids[raw_id]
            dir = os.path.join(self.data_dir, raw_id)
            self.images.extend([{"image": os.path.join(dir, image), "category": fine_id} for image in os.listdir(dir)])

    def transform(self, img):
        if self.mode == "train":
            img = random_sized_crop(
                im=img, size=TRAIN_IMAGE_SIZE, area_frac=0.08
            )
            img = horizontal_flip(im=img, p=0.5, order="HWC")
        else:
            img = scale(TEST_IMAGE_SIZE, img)
            img = center_crop(TRAIN_IMAGE_SIZE, img)
        img = img.transpose([2, 0, 1]) / 255
        if self.mode == "train":
            img = lighting(img, 0.1, np.array(EIGENVALUES), np.array(EIGENVECTORS))
        img = color_norm(img, [0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index]["image"])
        while img is None:
            index = index + 1
            img = cv2.imread(self.images[index % len(self.images)]['image'])
    
        img = img.astype(np.float32, copy=False)
        img = self.transform(img)
        category = self.images[index]["category"]
        return img, category

if __name__ == '__main__' :
    imagenet = Imagenet(root_dir='data', mode='train')
    training_params = {"batch_size": 32,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 12}
    training_generator = DataLoader(imagenet, **training_params)
    
    error_count = 0
    for i, (images, target) in enumerate(training_generator):
        print (target)
        break

    # path = 'data/ILSVRC2012_img_train/n02494079/n02494079_3206.JPEG'
    # img = cv2.imread(path)
    # img = img.astype(np.float32, copy=False)
