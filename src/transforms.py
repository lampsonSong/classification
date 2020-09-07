import math
import cv2
import numpy as np


def contrast_and_brightness(img, alpha=[0.5,1.5], beta=[-30,30]):
    """使用公式f(x)=α.g(x)+β"""
    #α调节对比度，β调节亮度
    a = np.random.uniform(alpha[0], alpha[1])
    b = math.ceil(np.random.uniform(beta[0],beta[1]))

    blank = np.zeros(img.shape,img.dtype)#创建图片类型的零矩阵
    dst = cv2.addWeighted(img, a, blank, 1-a, b)#图像混合加权
    return dst

def bgr_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return gray
        

def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")


def horizontal_flip(im, p, order="CHW"):
    """Performs horizontal flip (CHW or HWC format)."""
    assert order in ["CHW", "HWC"]
    if np.random.uniform() < p:
        if order == "CHW":
            im = im[:, :, ::-1]
        else:
            im = im[:, ::-1, :]
    return im


def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y : (y + size), x : (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop


def scale(size, im):
    """Performs scaling (HWC format)."""
    h, w = im.shape[:2]
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    h_new, w_new = size, size
    if w < h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)


def center_crop(size, im):
    """Performs center cropping (HWC format)."""
    h, w = im.shape[:2]
    y = int(math.ceil((h - size) / 2))
    x = int(math.ceil((w - size) / 2))
    im_crop = im[y : (y + size), x : (x + size), :]
    assert im_crop.shape[:2] == (size, size)
    return im_crop

def rotate_bound(angle, img):
    h,w = img.shape[:2]
    cx, cy = w // 2, h // 2

    rotate_matrix = cv2.getRotationMatrix2D((cx,cy), -angle, 1.0)
    cos = np.abs(rotate_matrix[0, 0])
    sin = np.abs(rotate_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotate_matrix[0, 2] += new_w / 2 - cx
    rotate_matrix[1, 2] += new_h / 2 - cy

    return cv2.warpAffine(img, rotate_matrix, (new_w, new_h))

def random_sized_crop(im, size, area_frac=0.08, max_iter=10):
    """Performs Inception-style cropping (HWC format)."""
    h, w = im.shape[:2]
    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h_crop = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            im_crop = im[y : (y + h_crop), x : (x + w_crop), :]
            assert im_crop.shape[:2] == (h_crop, w_crop)
            im_crop = cv2.resize(im_crop, (size, size), interpolation=cv2.INTER_LINEAR)
            return im_crop.astype(np.float32)
    return center_crop(size, scale(size, im))


def lighting(im, alpha_std, eig_val, eig_vec):
    """Performs AlexNet-style PCA jitter (CHW format)."""
    if alpha_std == 0:
        return im
    alpha = np.random.normal(0, alpha_std, size=(1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0), axis=1
    )
    for i in range(im.shape[0]):
        im[i] = im[i] + rgb[2 - i]
    return im

def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    x = np.random.uniform(-10, 10, 3) * [h_gain, s_gain, v_gain] + 1
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
    np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])  # inplace hue clip (0 - 179 deg)

    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def img_blur(img):
    radius = math.ceil(np.random.random()*20)
    if(np.random.random() < 0.5):
        img = cv2.blur(img, (radius, radius))
    return img

def norm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img[:,:,0] = (img[:,:,0] / 255. - mean[0]) / std[0]
    img[:,:,1] = (img[:,:,1] / 255. - mean[1]) / std[1]
    img[:,:,2] = (img[:,:,2] / 255. - mean[2]) / std[2]

    return img
