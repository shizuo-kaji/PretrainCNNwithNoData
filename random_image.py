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
from dataset import generate_random_image

def generate_and_save_random_image(idx, args=None):
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    sample = generate_random_image(args)
    sample.save(os.path.join(args.output,"{}{:0>8}.jpg".format(args.prefix,idx)))

#%%
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FractalDB Pre-training")
    # paths
    parser.add_argument("--output", '-o', default="random/train", type = str, help="path to trained weight")
    parser.add_argument("--prefix", '-pf', default="v", type = str, help="prefix to the filename")
    # image generator
    parser.add_argument("--alpha_range", '-ar', default=[0.01,1], type = float, nargs=2, help="")
    parser.add_argument("--beta_range", '-br', default=[0.5,2], type = float, nargs=2, help="")
    parser.add_argument("--n_samples", '-n', default=50000, type = int, help="number of images in an epoch")
    parser.add_argument("--prob_binary", '-pb', default=1.0, type = float, help="probability of binarising the generated image")
    parser.add_argument("--img_size", '-is', default=256, type = int, help="image size")
    parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers")
    parser.add_argument("--prob_colour", '-pc', default=1.0, type = float,  help="")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    task = partial(generate_and_save_random_image, args=args)
    if args.num_workers>1:
        pool = Pool(args.num_workers)
        #with ProcessPoolExecutor(args.batch) as executor:
        #    tqdm(executor.map(comp_PH,fns), total=len(fns))
        with tqdm(total=args.n_samples) as t:
            for _ in pool.imap_unordered(task,range(args.n_samples)):
                t.update(1)
        pool.close()
    else:
        for i in tqdm(range(args.n_samples)):   
            task(i)
