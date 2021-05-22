# -*- coding: utf-8 -*-
# Generating random images with a bump frequency profile
# By S. Kaji

#%%
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import threshold_otsu
from PIL import Image
import os
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from functools import partial

def dt(img):
    if img.max()==img.min():
        return(np.zeros_like(img))
    bw_img = (img >= threshold_otsu(img))
    dt_img = distance_transform_edt(bw_img)-distance_transform_edt(~bw_img)
    return(bw_img)
    #return(dt_img)

def generate_random_image(idx, args=None):
    sample=np.zeros(1)
    p = np.random.uniform(0,1)
    if p<args.prob_colour:
        channel = 3
    else:
        channel = 1
    while sample.max()-sample.min()<1e-10:
        sample = []
        for i in range(channel):
            alpha = np.random.uniform(*args.alpha_range)
            beta = np.random.uniform(*args.beta_range)
            x = np.linspace(1,np.exp(alpha)*args.img_size,args.img_size)
            X, Y = np.meshgrid(x,x)
            noise = np.random.uniform(0,1,(args.img_size,args.img_size))
            f = fft2(noise)
            f = f/(X**2+Y**2)**beta
            sample.append(ifft2(f).real)
        sample = np.array(sample)
    #print(sample.shape)
    #print(sample.min(), sample.max())
    for i in range(channel):
        p = np.random.uniform(0,1)
        if p<args.prob_binary/2:
            sample[i] = (sample[i] >= threshold_otsu(sample[i]))
        elif p<args.prob_binary:
            sample[i] = (sample[i] < threshold_otsu(sample[i]))
        else:
            sample[i] = (sample[i]-sample[i].min())/np.ptp(sample[i])
    sample = (255*sample).astype(np.uint8).transpose(1,2,0).squeeze()
    Image.fromarray(sample).save(os.path.join(args.output,"{}{:0>8}.jpg".format(args.prefix,idx)))

#%%
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FractalDB Pre-training")
    # paths
    parser.add_argument("--output", '-o', default="random/train", type = str, help="path to trained weight")
    parser.add_argument("--prefix", '-pf', default="v", type = str, help="prefix to the filename")
    # image generator
    parser.add_argument("--alpha_range", '-ar', default=[0.01,1], type = float, nargs=2, help="")
    parser.add_argument("--beta_range", '-br', default=[0.5,2], type = float, nargs=2, help="")
    parser.add_argument("--n_samples", '-n', default=200000, type = int, help="number of images in an epoch")
    parser.add_argument("--prob_binary", '-pb', default=1.0, type = float, help="probability of binarising the generated image")
    parser.add_argument("--img_size", '-is', default=256, type = int, help="image size")
    parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers")
    parser.add_argument("--prob_colour", '-pc', default=1.0, type = float,  help="")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    task = partial(generate_random_image, args=args)
    if args.num_workers>1:
        pool = Pool(args.num_workers)
        #with ProcessPoolExecutor(args.batch) as executor:
        #    tqdm(executor.map(comp_PH,fns), total=len(fns))
        with tqdm(total=args.n_samples) as t:
            for _ in pool.imap_unordered(task,range(args.n_samples)):
                t.update(1)
    else:
        for i in tqdm(range(args.n_samples)):   
            task(i)
# %%
# plt.imshow(noise, cmap='gray')
# #%%
# plt.imshow(dt(sample), cmap='gray')

# # %%
# plt.imshow(f.real)
# # %%
# f.shape, f.real.min(), f.real.max()
# # %%
