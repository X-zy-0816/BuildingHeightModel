#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import torch
import numpy as np
import rasterio as rio
from osgeo import gdal
from os.path import join
from ptsemseg.models import TLCNetU

# Related setting

device = 'cuda'
# GPU id
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# the size of each part of images
grid = 400
# the whole image used to predict
filelistnew = [r'/home/zhiyuan/files/BuildingHeight/fz/he_multi_cropped.tif']
# the path of pre-trained model
resume = r'./runs/tlcnetu_zy3bh/V1/finetune_598.tar'

# Setup Model
model = TLCNetU(n_classes=1).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at resume")
    print("=> Will start from scratch.")

model.eval()

def predict_whole_image(model, image, r, c, grid=400, overlap=50):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    overlap: overlap size for sliding window
    '''
    n, b, rows, cols = image.shape
    res = np.zeros((rows, cols), dtype=np.float32)
    seg = np.zeros((rows, cols), dtype=np.uint8)

    stride = grid - overlap  # Stride is grid size minus overlap
    num_patch = len(range(0, rows - grid, stride)) * len(range(0, cols - grid, stride))
    print('num of patch is', num_patch)
    k = 0
    for i in range(0, rows - grid, stride):
        for j in range(0, cols - grid, stride):
            patch = image[0:, 0:, i:i + grid, j:j + grid].astype('float32')
            if np.max(patch.flatten()) <= 10e-8:
                continue
            start = time.time()
            patch = torch.from_numpy(patch).float()

            # normalization
            patch = patch / 1000.0

            # Model prediction
            pred = model(patch.to(device))
            pred0 = pred[0].cpu().detach().numpy()  # height
            pred1 = pred[1].cpu().detach().numpy()  # seg

            # Update results with overlap handling
            res[i:i + grid, j:j + grid] = np.maximum(res[i:i + grid, j:j + grid], np.squeeze(pred0))
            seg[i:i + grid, j:j + grid] = np.argmax(np.maximum(seg[i:i + grid, j:j + grid], np.squeeze(pred1)), axis=0)

            end = time.time()
            k += 1
            # print('patch [%d/%d] time elapse:%.3f' % (k, num_patch, (end-start)))

    res = res[0:r, 0:c].astype(np.float32)
    seg = seg[0:r, 0:c].astype(np.uint8)
    return res, seg

# read all city names
def read_filepath(filepath, filename):
    filelist = list()
    for root, dirs, files in os.walk(filepath):
        for name in files:
            if name.endswith(filename):
                filelist.append(join(root, name))
    filelist.sort()
    return filelist


def loadenvi(file, band=1):
    dataset = gdal.Open(file)
    band = dataset.GetRasterBand(band)
    data = band.ReadAsArray()
    dataset = None
    band = None
    return data

data = loadenvi(filelistnew[0], band=1)
print(data.shape)
print(data.dtype)
data = None

for file in filelistnew[:1]:
    # filepath
    idirname = os.path.dirname(file)
    predpath = join(idirname, 'pred')
    respath = join(predpath, 'predtlcnetu_200.tif')  # model tlcnetu, epoch 200
    segpath = join(predpath, 'predtlcnetu_200_seg.tif')
    if os.path.exists(respath):
        print('file: %s already exist, then skip' % respath)
        continue
    if not os.path.exists(predpath):
        print('mkdir %s' % predpath)
        os.mkdir(predpath)

    print('process: %s', file)
    nadpath = join(idirname, 'nads.tif')
    fwdpath = join(idirname, 'fwds.tif')
    bwdpath = join(idirname, 'bwds.tif')
    # 1.read image
    band1 = loadenvi(file, band=1)
    r, c = band1.shape[:2]
    img = np.zeros((r, c, 7), dtype='uint16')
    for i in range(4):
        img[:, :, i] = loadenvi(file, i + 1)
    img[:, :, 4] = loadenvi(nadpath)
    img[:, :, 5] = loadenvi(fwdpath)
    img[:, :, 6] = loadenvi(bwdpath)
    img = img.transpose(2, 0, 1)  # C H W

    # img=img/10000
    img = np.expand_dims(img, axis=0)  # 1 C H W

    rows = math.ceil(r / grid) * grid
    cols = math.ceil(c / grid) * grid
    img = np.pad(img, ((0, 0), (0, 0), (0, rows - r), (0, cols - c)), 'symmetric')

    # 2. predict
    starttime = time.time()
    res = predict_whole_image(model, img, r, c, grid=400)
    endtime = time.time()
    print('success: %s, time is %f' % (file, endtime - starttime))

    img = 0  # release

    # 3. export
    reffile = file
    rastermeta = rio.open(reffile).profile
    rastermeta.update(dtype=res[0].dtype, count=1, compress='lzw')
    with rio.open(respath, mode="w", **rastermeta) as dst:
        dst.write(res[0], 1)
    rastermeta.update(dtype=res[1].dtype, count=1, compress='lzw')
    with rio.open(segpath, mode="w", **rastermeta) as dst:
        dst.write(res[1], 1)
#     tif.imsave(respath, res[0]) # building height
#     tif.imsave(segpath, res[1]) # building segmentation

