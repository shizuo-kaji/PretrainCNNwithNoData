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
import os,json
from tqdm import tqdm
import argparse
from multiprocessing import Pool
from functools import partial

def generate_and_save_random_image(idx, args=None, outdir=None, prefix=""):
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    sample = generate_random_image(args)
    sample.save(os.path.join(outdir,"{}{:0>8}.jpg".format(prefix,idx)))

def generate_random_image(args, require_at_least_one_binary=False):
    np.random.seed()
    sample=np.zeros(1)
    channel = 3
    while sample.max()-sample.min()<1e-10:
        sample = []
        for i in range(channel):
            alpha = np.random.uniform(*args.alpha_range)
            beta = np.random.uniform(*args.beta_range)
            x = np.linspace(1,1+alpha*args.img_size,args.img_size)
            X, Y = np.meshgrid(x,x)
            noise = np.random.uniform(0,1,(args.img_size,args.img_size))
            f = fft2(noise)
            #f = noise
            f = f/(X**2+Y**2)**beta
            sample.append(ifft2(f).real)
        sample = np.array(sample)
    #print(sample.shape)
    #print(sample.min(), sample.max())
    at_least_one_binary = not require_at_least_one_binary
    for i in range(channel):
        p = np.random.uniform(0,1)
        if p<args.prob_binary/2:
            sample[i] = (sample[i] >= threshold_otsu(sample[i]))
            at_least_one_binary=True
        elif p<args.prob_binary:
            sample[i] = (sample[i] < threshold_otsu(sample[i]))
            at_least_one_binary=True
        else:
            sample[i] = (sample[i]-sample[i].min())/np.ptp(sample[i])
    if not at_least_one_binary:
        i = np.random.randint(channel)
        sample[i] = (sample[i] >= threshold_otsu(sample[i]))

    sample = Image.fromarray((255*sample).astype(np.uint8).transpose(1,2,0).squeeze())

    p = np.random.uniform(0,1)
    if p>args.prob_colour:
        sample = sample.convert('L')
    return sample


#%%
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="generate random images in the fourier domain")
    # paths
    parser.add_argument("--output", '-o', default=None, type = str, help="path to output images")
    parser.add_argument("--prefix", '-pf', default="", type = str, help="prefix to the filename")
    # image generator
    parser.add_argument("--alpha_range", '-ar', default=[0,1], type = float, nargs=2, help="")
    parser.add_argument("--beta_range", '-br', default=[1,2], type = float, nargs=2, help="")
    parser.add_argument("--n_samples", '-n', default=200000, type = int, help="number of images")
    parser.add_argument("--n_samples_val", '-nv', default=5000, type = int, help="number of images for validation")
    parser.add_argument("--img_size", '-is', default=256, type = int, help="image size")
    parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers")
    parser.add_argument("--prob_colour", '-pc', default=0.5, type = float,  help="")
    parser.add_argument("--prob_binary", '-pb', default=0.5, type = float, help="probability of binarising the generated image")
    args = parser.parse_args()

    if args.output is None:
        args.output = "random{}kpb{}pc{}a{}-{}b{}-{}".format(args.n_samples//1000,args.prob_binary,args.prob_colour,args.alpha_range[0],args.alpha_range[1],args.beta_range[0],args.beta_range[1])

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    if args.n_samples_val > 0:
        phases = ["train","val"]
        prefix = ["t","v"]
        nsmp = [args.n_samples,args.n_samples_val]
    else:
        phases = ["."]
        prefix = [args.prefix]
        nsmp = [args.n_samples]
    for k,phase in enumerate(phases):
        n = nsmp[k]
        outdir = os.path.join(args.output,phase)
        os.makedirs(outdir, exist_ok=True)
        task = partial(generate_and_save_random_image, args=args, outdir=outdir, prefix=prefix[k])
        if args.num_workers>1:
            pool = Pool(args.num_workers)
            #with ProcessPoolExecutor(args.batch) as executor:
            #    tqdm(executor.map(comp_PH,fns), total=len(fns))
            with tqdm(total=n, ncols=100, ascii=True) as t:
                for _ in pool.imap_unordered(task,range(n)):
                    t.update(1)
            pool.close()
        else:
            for i in tqdm(range(n), ncols=100, ascii=True):   
                task(i)
