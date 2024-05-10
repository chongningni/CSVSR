import pytorch_ssim
import torch.nn as nn
import torch
from torch.autograd.variable import Variable
import numpy as np
import torch.nn.functional as F


def get_loss_fn(model_name):

    if model_name in ['EDVR','MYEDVR','MYEDVR_UP1','MYEDVR_UP2','MYEDVR_UP4','MYEDVR_SAconv','MYEDVR_Rout','MYEDVR_Mask','MYEDVR_All','MYEDVR_Res','MYEDVR_All_Res','STAN','MYEDVR_Rout_Res']:
        return EDVR_loss() 
    elif model_name in ['VSRNet','SOFVSR','CFSRCNN','D3DNet']:
        return nn.MSELoss()
    else:
        return nn.L1Loss()

def OFR_loss(x0, x1, optical_flow):
    warped = optical_flow_warp(x0, optical_flow)
    loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)
    return loss

def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))

def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    b, _ , h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1),1)
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border')
    return output

class MSE_and_SSIM_loss(nn.Module):
    def __init__(self, alpha=0.9):
        super(MSE_and_SSIM_loss, self).__init__()
        self.MSE = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.alpha = alpha

    def forward(self, img1, img2):
        loss = self.alpha*self.MSE(img1, img2) + (1 - self.alpha)*(1 - self.SSIM(img1, img2))
        return loss


class L1andAlignLoss(nn.Module):
    def __init__(self,):
        super(L1andAlignLoss, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, img1, img2):
        loss = self.L1(img1, img2)
        return loss

class EDVR_loss(nn.Module):
    def __init__(self,):
        super(EDVR_loss, self).__init__()
        self.L1 = nn.MSELoss()

    def forward(self, img1, img2):
        loss = (self.L1(img1, img2) + 1e-6).sqrt()
        return loss
