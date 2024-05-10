import argparse
import sys
import scipy
import os
from PIL import Image
import torch
import numpy as np
from skimage import io, transform
from model import ModelFactory
from torch.autograd import Variable
import time
from VSR_datasets import VSR_AsyTestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim
from solver import chop_forward, rgb2ycbcrT
description='Video Super Resolution pytorch implementation'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _comput_PSNR(input, target):
        """Compute PSNR between two image array and return the psnr summation"""
        shave = 4
        ch, h, w = input.size()
        input_Y = rgb2ycbcrT(input.cpu())
        target_Y = rgb2ycbcrT(target.cpu())
        diff = (input_Y - target_Y).view(1, h, w)

        diff = diff[:, shave:(h - shave), shave:(w - shave)]
        mse = diff.pow(2).mean()
        psnr = -10 * np.log10(mse)
        return psnr

def forward_x8(lr, forward_function=None):
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            #print(v2np.shape)
            if op == 'v':
                tfnp = v2np[:, :, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 2, 4, 3)).copy()
	
            ret = Variable(torch.Tensor(tfnp).cuda())
            #ret = ret.half()

            return ret

        def _transform_back(v, op):
       		
            if op == 'v':
                tfnp = v[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v.transpose((0, 1, 3, 2)).copy()
	
            return tfnp

        
        x = [lr]
        for tf in 'v', 'h': x.extend([_transform(_x, tf) for _x in x])
       
        list_r = []
        for k in range(len(x)):
            z = x[k]
            r, _ = forward_function(z)
            r = r.data.cpu().numpy()
            if k % 4 > 1:
                    r =  _transform_back(r, 'h')
            if (k % 4) % 2 == 1:
                    r =  _transform_back(r, 'v')
            list_r.append(r)
        y = np.sum(list_r,  axis=0)/4.0
       
        y = Variable(torch.Tensor(y).cuda())
        if len(y) == 1: y = y[0]
        return y
        
def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture.')
parser.add_argument('-s', '--scales', metavar='S', type=str, default='2,3,4', 
                    help='interpolation scale. Default 4')
parser.add_argument('-t', '--test-set', metavar='NAME', type=str, default='001',
                    help='dataset for testing.')
parser.add_argument('-mp', '--model-path', metavar='MP', type=str, default='model',
                    help='model path.')
parser.add_argument('-sp', '--save-path', metavar='SP', type=str, default='res',
                    help='saving directory path.')
args = parser.parse_args()

model_factory = ModelFactory()
model = model_factory.create_model(args.model, scale = args.scales)
model_name = model.name 
# model_path = os.path.join(args.model_path, 'best_model.pt')
model_path = args.model_path
if not os.path.exists(model_path):
    raise Exception('Cannot find %s.' %model_path)
model = torch.load(model_path)
model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

test_datasets = (
        [_ for _ in args.test_set.split(",")] if args.test_set else ["001"]
    )

path = args.save_path
if not os.path.exists(path):
    os.makedirs(path)

scales = args.scales.split(",") if args.scales else [2, 3, 4]
# scales = [float(s) for s in scales]

for dataset_name in test_datasets:
    for scale in scales:

        scale = scale.split("/") if scale else [2, 3, 4]
        scale = [float(s) for s in scale]

        rslt_path = os.path.join(path, "results", dataset_name, model_name, "x" + str(scale[0]) + "/" + str(scale[1]))
        if not os.path.exists(rslt_path):
            os.makedirs(rslt_path)
        
        print("==== Dataset {}, Scale Factor x{:.2f} x{:.2f} ====".format(dataset_name, scale[0], scale[1]))
        test_path = os.path.join('data/benchmarks/', dataset_name)
        val_dataset  = VSR_AsyTestDataset(dir=test_path, scale = scale, model_name = model_name)
        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        psnrs, ssims, run_times, losses = [], [], [], []
        avr_psnr = 0
        avr_ssim = 0
        for batch, sample in tqdm(enumerate(dataloader), total=len(val_dataset)):
            input_batch, label_batch, name, scale_h, scale_w = sample['lr'], sample['hr'], sample['im_name'], sample['scale_h'], sample['scale_w']
            input_batch, label_batch = (Variable(input_batch.cuda()), Variable(label_batch.cuda()))
            torch.cuda.synchronize()
            start_time = time.time()
            if model_name in ['MYEDVR','EDVR']:
                output_batch = model(input_batch, [torch.Tensor([scale_h]), torch.Tensor([scale_w])])
            else:
                output_batch, _ = model(input_batch, [torch.Tensor([scale_h]), torch.Tensor([scale_w])])
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(elapsed_time)
            ssim = pytorch_ssim.ssim(output_batch + 0.5, label_batch + 0.5, size_average=False)
            ssim = torch.sum(ssim.data)
            avr_ssim += ssim
            # calculate PSRN
            output = output_batch.data
            label = label_batch.data

            output = (output + 0.5) * 255
            label = (label + 0.5) * 255

            output = quantize(output, 255)
            label = quantize(label, 255)
            # diff = input - target

            output = output.squeeze(dim=0)
            label = label.squeeze(dim=0)

            psnr = _comput_PSNR(output / 255.0, label / 255.0)
            # print("psnr:",psnr.item())
            # print(111111111111111111111,psnr)
            avr_psnr += psnr

            # save psnrs and outputs for statistics and generate image at test time
            psnrs.append(psnr)
            ssims.append(ssim)
            run_times.append(elapsed_time)

            #write image
            img_name = os.path.join(rslt_path, name[0])
            Image.fromarray(np.around(output.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)).save(img_name)
            

        epoch_size = len(val_dataset)
        avr_psnr /= epoch_size
        avr_ssim /= epoch_size
        mean_runtime = np.array(run_times).mean()
        print("- PSNR: {:.4f}".format(avr_psnr))
        print("- SSIM: {:.4f}".format(avr_ssim))
        print("- Runtime : {:.4f}".format(mean_runtime))
        print("Finished!")
        print("===================================================")



        
