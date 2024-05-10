from __future__ import division
import torchvision
import torchvision.transforms as T
import os
import glob
import scipy.misc
import scipy.ndimage
from PIL import Image, ImageFile
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
# import skimage.transform as sktransform
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
import torchvision.transforms.functional as TF


ImageFile.LOAD_TRUNCATED_IMAGES = True

class VSR_TrainDataset(object):
    def __init__(self, dir, lr_img_sz, scale, model_name):

        self.dir_HR = os.path.join(dir, "HR")
        self.lis = sorted(os.listdir(self.dir_HR))

        self.lr_img_sz = lr_img_sz
        self.center = 2
        if model_name == 'SOFVSR':
            self.ToTensor = VToTensor_SOFVSR()
        else:
            self.ToTensor = VToTensor()
        self.scales = scale
        self.scale = 4
        self.scale_another = 3
        self.min_scale = 1
        self.max_scale = 5
        self.index = 0
        self.model_name = model_name

    def __len__(self):
        return len(self.lis)

    def __getitem__(self, idx):
        HR = os.path.join(self.dir_HR, self.lis[idx])

        if len(self.scales) == 1:
            self.scale = int(self.scales[0])
            self.scale_another = self.scale
        elif self.model_name in ['CSVSR']:
            if self.index % 32 == 0 and self.index < 400:
                scale_width = random.uniform(1,5)
                self.scale = scale_width
                self.scale_another = scale_width
            elif self.index % 32 == 0 and self.index < 600:
                scale_width = random.uniform(2,5)
                self.scale = scale_width
                self.scale_another = scale_width
            elif self.index % 32 == 0 and self.index < 801:
                scale_width = random.uniform(3,5)
                self.scale = scale_width
                self.scale_another = scale_width
   
        ims = sorted(os.listdir(HR))
        # get frame size
        image = io.imread(os.path.join(HR, ims[0]))
        row, col, ch = image.shape
        frames_hr = np.zeros((5, row, col , ch))
        if len(ims) > 5:
            center = random.randint(2, len(ims)-3)
        else:
            center = len(ims) // 2
        
        self.center = center

        for j in range(center - 2, center + 3):  # only use 5 frames
            i = j - center + 2
            frames_hr[i, :, :, :] = io.imread(os.path.join(HR, ims[j]))

        lr, hr = self._transform(frames_hr, self.scale, self.scale_another)

        sample = {'lr': lr, 'hr': hr, 'im_name': ims[center], 'scale': self.scale, 'scale_another': self.scale_another}

        sample = self.ToTensor(sample)

        self.index = self.index + 1

        return sample

    def _transform(self, x, scale_width, scale_length = 4):
        lr_img_sz = self.lr_img_sz
        hr_img_sz = int(lr_img_sz * scale_width)
        hr_img_sz_another = int(lr_img_sz * scale_length)
        
        transform = transforms.Compose([
            VRandomCrop(hr_img_sz),
            VDataAug(),
            lambda x : [[skimage.transform.resize(_, (lr_img_sz, lr_img_sz), order=3) for _ in x], x],
        ])

        transform_another = transforms.Compose([
            VRandomCrop((hr_img_sz, hr_img_sz_another)),
            VDataAug_another(),
            lambda x : [[skimage.transform.resize(_, (lr_img_sz, lr_img_sz), order=3) for _ in x], x],
        ])

        if scale_width == scale_length:
            lr, hr = transform(x)
        else:
            lr, hr = transform_another(x)

        lr = np.array(lr)
        if self.model_name == 'SOFVSR':
            hr = hr[1:4, :, :, :]
            lr = lr[1:4, :, :, :]
        else:
            hr = hr[self.center]
        return lr, hr


class VSR_TestDataset(object):
    def __init__(self, dir, scale, model_name):

        self.dir_HR = os.path.join(dir, "HR")
        self.lis = sorted(os.listdir(self.dir_HR))

        self.center = 2
        if model_name == 'bicubic':
            self.VToTensor = VToTensor_bicubic()
        elif model_name == 'SOFVSR':
            self.VToTensor = VToTensor_SOFVSR_test()
        else:
            self.VToTensor = VToTensor_test()

        self.scales = scale
        self.scale = 4.0
        self.model_name = model_name

    def __len__(self):
        return len(self.lis)

    def __getitem__(self, idx):
        HR = os.path.join(self.dir_HR, self.lis[idx])

        if isinstance(self.scales, list):
            self.scale = self.scales[-1]
        else:
            self.scale = self.scales
        
        ims = sorted(os.listdir(HR))
        # get frame size
        image = io.imread(os.path.join(HR, ims[0]))
        row, col, ch = image.shape
        frames_hr = np.zeros((5, row, col , ch))
        if len(ims) > 5:
            center = random.randint(2, len(ims)-3)
        else:
            center = len(ims) // 2
        
        self.center = center

        for j in range(center - 2, center + 3):  # only use 5 frames
            i = j - center + 2
            frames_hr[i, :, :, :] = io.imread(os.path.join(HR, ims[j]))

        lr, hr = self.continuous_transform(frames_hr, self.scale)

        sample = {'lr': lr, 'hr': hr, 'im_name': ims[center], 'scale': self.scale}

        sample = self.VToTensor(sample)

        return sample

    def _transform(self, x, scale_width, scale_length=2):

        n, row, col, ch = x.shape
        lr_img_sz_w = int(row/scale_width)
        lr_img_sz_h = int(col/scale_width)

        hr_img_sz_w = int(lr_img_sz_w * scale_width)
        hr_img_sz_h = int(lr_img_sz_h * scale_width)

        x = x[:,0:hr_img_sz_w,0:hr_img_sz_h,:]
        transform = transforms.Compose([
            lambda x : [[skimage.transform.resize(_, (lr_img_sz_w, lr_img_sz_h), order=3) for _ in x], x],
        ])
        lr, hr = transform(x)
        lr = np.array(lr)
        # hr = hr[self.center]
        if self.model_name == 'SOFVSR':
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_w,0:hr_img_sz_h,:]
            lr = lr[1:4, :, :, :]
        else:
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_w,0:hr_img_sz_h,:]
        return lr, hr

    def continuous_transform(self, x, scale_width, scale_length=2):

        n, row, col, ch = x.shape
        lr_img_sz_w = int((row//(scale_width*4))*4)
        lr_img_sz_h = int((col//(scale_width*4))*4)

        hr_img_sz_w = int(lr_img_sz_w * scale_width)
        hr_img_sz_h = int(lr_img_sz_h * scale_width)

        x = x[:,0:hr_img_sz_w,0:hr_img_sz_h,:]
        transform = transforms.Compose([
            lambda x : [[skimage.transform.resize(_, (lr_img_sz_w, lr_img_sz_h), order=3) for _ in x], x],
        ])
        lr, hr = transform(x)
        lr = np.array(lr)
        # hr = hr[self.center]
        if self.model_name == 'SOFVSR':
            # hr = hr[1:4, :, :, :]
            # hr = hr[1, :, :, :]
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_w,0:hr_img_sz_h,:]
            lr = lr[1:4, :, :, :]
        else:
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_w,0:hr_img_sz_h,:]
        return lr, hr


class VDataAug(object):
    def __call__(self, hr):

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5

        num, r, c, ch = hr.shape

        if hflip:
            for idx in range(num):
                hr[idx, :, :, :] = hr[idx, :, ::-1, :]
        if vflip:
            for idx in range(num):
                hr[idx, :, :, :] = hr[idx, ::-1, :, :]
        if rot90:
            hr = hr.transpose(0, 2, 1, 3)

        return hr

class VDataAug_another(object):
    def __call__(self, hr):

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        num, r, c, ch = hr.shape

        if hflip:
            for idx in range(num):
                hr[idx, :, :, :] = hr[idx, :, ::-1, :]
        if vflip:
            for idx in range(num):
                hr[idx, :, :, :] = hr[idx, ::-1, :, :]

        return hr


class VRandomCrop(object):

    """crop randomly the image in a sample
    Args, output_size:desired output size. If int, square crop is mad

    """
    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, hr):

        h, w = hr.shape[1: 3]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        new_hr = hr[:,top:top + new_h, left: left + new_w, :]

        return new_hr

class VToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale, scale_another = sample['lr']/255.0 - 0.5, sample['hr']/255.0 - 0.5, sample['im_name'], sample['scale'], sample['scale_another']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'im_name':name, 'scale':scale, 'scale_another':scale_another}

class VToTensor_test(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale = sample['lr']/255.0 - 0.5, sample['hr']/255.0 - 0.5, sample['im_name'], sample['scale']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'im_name':name, 'scale':scale}
    
class VToTensor_asytest(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale_h, scale_w = sample['lr']/255.0 - 0.5, sample['hr']/255.0 - 0.5, sample['im_name'], sample['scale_h'], sample['scale_w']
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'im_name':name, 'scale_h':scale_h, 'scale_w':scale_w}

class VToTensor_bicubic(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale = sample['lr']/255.0 - 0.5, sample['hr']/255.0 - 0.5, sample['im_name'], sample['scale_h']
        w, h, c = hr.shape
        # torch.cuda.synchronize()
        start_time = time.time()
        hr_bicubic = skimage.transform.resize(lr[2], (w, h), order=3)    
        # torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        hr_bicubic = torch.from_numpy(hr_bicubic).float()
        return {'lr': lr.permute(0, 3, 1, 2), 'hr': hr.permute(2, 0, 1), 'hr_bicubic': hr_bicubic.permute(2, 0, 1), 'im_name':name, 'scale':scale, 'elapsed_time':elapsed_time}

class VToTensor_SOFVSR(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale, scale_another = sample['lr']/255.0, sample['hr']/255.0, sample['im_name'], sample['scale'], sample['scale_another']

        lr_y, _, _ = rgb2ycbcr(lr) 
        hr_y, _, _ = rgb2ycbcr(hr) 

        lr = torch.from_numpy(lr_y).float()
        hr = torch.from_numpy(hr_y).float()

        n_frames, h_lr, w_lr = lr.size()
        lr = lr.view(-1, 1, h_lr, w_lr)

        n_frames, h_hr, w_hr = hr.size()
        hr = hr.view(-1, 1, h_hr, w_hr)
        # lr = torch.unsqueeze(lr, dim=1)
        # hr = torch.unsqueeze(hr, dim=1)

        return {'lr': lr, 'hr': hr, 'im_name':name, 'scale':scale, 'scale_another':scale_another}

def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, :, 0] + 0.504 * img_rgb[:, :, :, 1] + 0.098 * img_rgb[:, :, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, :, 0] - 0.291 * img_rgb[:, :, :, 1] + 0.439 * img_rgb[:, :, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, :, 0] - 0.368 * img_rgb[:, :, :, 1] - 0.071 * img_rgb[:, :, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


class VToTensor_SOFVSR_test(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lr, hr, name, scale = sample['lr']/255.0, sample['hr']/255.0, sample['im_name'], sample['scale']
        w, h, c = hr.shape
        hr_bicubic = skimage.transform.resize(lr[1], (w, h), order=3) 

        lr_y, _, _ = rgb2ycbcr(lr) 
        _, hr_bicubic_cb, hr_bicubic_cr = rgb2ycbcr_test(hr_bicubic) 

        lr = torch.from_numpy(lr_y).float()
        hr = torch.from_numpy(hr).float()
        lr = torch.unsqueeze(lr, dim=1)

        return {'lr': lr, 'hr': hr.permute(2, 0, 1), 'im_name':name, 'scale':scale, 'hr_bicubic_cb':hr_bicubic_cb, 'hr_bicubic_cr':hr_bicubic_cr}

def rgb2ycbcr_test(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr



class VSR_AsyTestDataset(object):
    def __init__(self, dir, scale, model_name):

        self.dir_HR = os.path.join(dir, "HR")
        self.lis = sorted(os.listdir(self.dir_HR))

        self.center = 2
        if model_name == 'bicubic':
            self.VToTensor = VToTensor_bicubic()
        else:
            self.VToTensor = VToTensor_asytest()

        self.scales = scale
        self.scale = 4.0
        self.model_name = model_name

    def __len__(self):
        return len(self.lis)

    def __getitem__(self, idx):
        HR = os.path.join(self.dir_HR, self.lis[idx])

        if isinstance(self.scales, list):
            self.scale_h = self.scales[0]
            self.scale_w = self.scales[1]
        else:
            self.scale_h = self.scales
            self.scale_w = self.scales
        
        ims = sorted(os.listdir(HR))
        # get frame size
        image = io.imread(os.path.join(HR, ims[0]))
        row, col, ch = image.shape
        frames_hr = np.zeros((5, row, col , ch))
        if len(ims) > 5:
            center = random.randint(2, len(ims)-3)
        else:
            center = len(ims) // 2
        
        self.center = center

        for j in range(center - 2, center + 3):  # only use 5 frames
            i = j - center + 2
            frames_hr[i, :, :, :] = io.imread(os.path.join(HR, ims[j]))

        lr, hr = self.continuous_transform(frames_hr, self.scale_h, self.scale_w)

        if self.scale_h==self.scale_w:
            sample = {'lr': lr, 'hr': hr, 'im_name': ims[center], 'scale_h': self.scale_h}
        else:
            sample = {'lr': lr, 'hr': hr, 'im_name': ims[center], 'scale_h': self.scale_h, 'scale_w': self.scale_w}

        sample = self.VToTensor(sample)

        return sample

    def _transform(self, x, scale_hight, scale_width=2):

        n, row, col, ch = x.shape
        lr_img_sz_h = int(row/scale_hight)
        lr_img_sz_w = int(col/scale_width)

        hr_img_sz_h = int(lr_img_sz_h * scale_hight)
        hr_img_sz_w = int(lr_img_sz_w * scale_width)

        x = x[:,0:hr_img_sz_h,0:hr_img_sz_w,:]
        transform = transforms.Compose([
            lambda x : [[skimage.transform.resize(_, (lr_img_sz_h, lr_img_sz_w), order=3) for _ in x], x],
        ])
        lr, hr = transform(x)
        lr = np.array(lr)
        # hr = hr[self.center]
        if self.model_name == 'SOFVSR':
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_h,0:hr_img_sz_w,:]
            lr = lr[1:4, :, :, :]
        else:
            hr = hr[self.center]
            hr = hr[0:hr_img_sz_h,0:hr_img_sz_w,:]
        return lr, hr

    def continuous_transform(self, x, scale_hight, scale_width=2):

        n, row, col, ch = x.shape
        lr_img_sz_h = int((row//(scale_hight*4))*4)
        lr_img_sz_w = int((col//(scale_width*4))*4)

        hr_img_sz_h = int(lr_img_sz_h * scale_hight)
        hr_img_sz_w = int(lr_img_sz_w * scale_width)

        x = x[:,0:hr_img_sz_h,0:hr_img_sz_w,:]
        transform = transforms.Compose([
            lambda x : [[skimage.transform.resize(_, (lr_img_sz_h, lr_img_sz_w), order=3) for _ in x], x],
        ])
        lr, hr = transform(x)
        lr = np.array(lr)
        hr = hr[self.center]
        hr = hr[0:hr_img_sz_h,0:hr_img_sz_w,:]
        
        return lr, hr
