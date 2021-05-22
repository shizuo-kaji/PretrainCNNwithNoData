# -*- coding: utf-8 -*-
# Computing Persistent Homology and its histogram
# By S. Kaji

import os,glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import argparse,json
import cripser
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import threshold_otsu
from PIL import Image
from multiprocessing import Pool
from functools import partial

def dt(img, binarize=True):
    if img.max()==img.min():
        return(np.zeros_like(img))
    if binarize:
        bw_img = (img >= threshold_otsu(img))
    else:
        bw_img = img
    dt_img = distance_transform_edt(bw_img)-distance_transform_edt(~bw_img)
    return(dt_img)

def comp_PH(img, distance_transform=True, binarize=True):
    #im = np.array(Image.open(fn).convert('L'),dtype=np.float64)
    if len(img.shape)>2:
        im = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        im = img
    if distance_transform:
        im = dt(im, binarize)
    pd = cripser.computePH(im)
    #pd = cripser.computePH(im,maxdim=args.maxdim,top_dim=args.top_dim,embedded=args.embedded)
    #print(sum(pd[:,0] == 1))
    #print(im.shape)
    return(pd)

def comp_save_PH(fname, PHdir=None):
    bfn = os.path.splitext(os.path.basename(fname))[0]
    sample = np.array(Image.open(fname).convert('L'),dtype=np.float64)
    ph = comp_PH(sample)
    np.save(os.path.join(PHdir, bfn), ph)

def comp_save_persistence_image(fname, args=None):
    bfn = os.path.splitext(os.path.basename(fname))[0]
    ph = np.load(fname)
    s = int((2*args.max_birth)/10)
    pim = persim.PersistenceImager(birth_range=(-args.max_birth,args.max_birth), pers_range=(0,args.max_life),pixel_size=s,kernel_params={'sigma': [[args.sigma, 0.0], [0.0, args.sigma]]})
    pims = []
    for d in [0,1]:
        p = (ph[ph[:,0]==d])[:,1:3]
        life = p[:,1]-p[:,0]
        life = np.clip(life,a_min=None,a_max=args.max_life)
        p[:,1] = life
        pims.append(np.sqrt(np.abs(pim.transform(p).ravel())).astype(np.float32))
        #print(pims[-1].max())
    #print(bfn, np.concatenate(pims).shape)
    np.save(os.path.join(args.output, bfn), np.concatenate(pims))

def hist_PH(ph, ls0, ls1, num_bins, max_life, max_birth, bandwidth=1):
    pds =[ph[ph[:,0] == i, 1:3] for i in range(2)]
    #print(len(pds[0]),len(pds[1]))
    life = [pd[:,1]-pd[:,0] for pd in pds]
    life = [l[l<1e+10] for l in life]  # remove permanent cycle
    life = [np.clip(l,0,max_life) for l in life]
    # hsb = np.zeros(args.num_bins[2])
    # for k,ind in enumerate(np.searchsorted(lsb,pds[1][:,0])):
    #     hsb[ind] += (pds[1][k,1]-pds[1][k,0]) ## lifetime weighted count
    # hs0 = gaussian_kde(life[0])(ls) * len(life[0])
    # hs1 = gaussian_kde(life[1])(ls) * len(life[1])
    hsl0, _ = np.histogram(life[0],bins=num_bins[0], range=(0,max_life))
    hsl1, _ = np.histogram(life[1],bins=num_bins[1], range=(0,max_life))
    hsb0, _ = np.histogram(pds[0][:,0],bins=num_bins[2], range=(-max_birth,0), weights=pds[0][:,1]-pds[0][:,0])
    hsb1, _ = np.histogram(pds[1][:,0],bins=num_bins[3], range=(-max_birth,max_birth), weights=pds[1][:,1]-pds[1][:,0])
    hsl0 = kern_smooth(hsl0, bandwidth=bandwidth, kern='hanning')*(1+ls0) # lifetime weighted
    hsl1 = kern_smooth(hsl1, bandwidth=bandwidth, kern='hanning')*(1+ls1)
    hsb0 = kern_smooth(hsb0, bandwidth=bandwidth, kern='hanning')
    hsb1 = kern_smooth(hsb1, bandwidth=bandwidth, kern='hanning')
    hsl0,hsl1,hsb0,hsb1 = np.log(hsl0+1), np.log(hsl1+1), np.log(hsb0+1)/100, np.log(hsb1+1)
    # print(np.min(pds[0][:,0]),np.max(pds[0][:,0]),np.min(pds[1][:,0]),np.max(pds[1][:,0]))
    # print(np.min(pds[0][:,1]),np.max(pds[0][:,1]),np.min(pds[1][:,1]),np.max(pds[1][:,1]),"\n")
    return(np.concatenate([hsl0,hsl1,hsb0,hsb1]))


# kern = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
def kern_smooth(y, bandwidth=11, kern='flat'):
    if bandwidth<2:
        return(y)
    b = int(bandwidth)
    if kern == 'flat':
        w=np.ones(bandwidth,'d')
    else:
        w=getattr(np,kern)(b)
    res = np.convolve(w/w.sum(),np.r_[y[b-1:0:-1],y,y[-2:-b-1:-1]],mode='valid')
    c = (len(res)-len(y))//2
    return(res[c:(c+len(y))])

if __name__== "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('target_dir',type=str)
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--max_life', '-ml', type=int, default=50)
    parser.add_argument('--max_birth', '-mb', type=int, default=50)
    parser.add_argument('--num_bins', '-n', type=int, nargs=3, default=[50,50,50,50])
    parser.add_argument('--bandwidth', '-b', type=int, default=1)
    parser.add_argument('--sigma', '-s', type=float, default=100)
    parser.add_argument('--imgtype', '-it', type=str, default="npy")
    parser.add_argument('--compute_hist', '-ch', action="store_true")
    parser.add_argument('--compute_persistence_image', '-cpi', action="store_true")
    parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers (data_loader)")
    args = parser.parse_args()
    
    target_dir = args.target_dir
    PHdir = args.output
    os.makedirs(PHdir, exist_ok=True)

    with open(os.path.join(PHdir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    fns=glob.glob(os.path.join(target_dir,"**/*.{}".format(args.imgtype)), recursive=True)

    if args.compute_hist:
        meanPHl0 = np.zeros(args.num_bins[0])
        meanPHl1 = np.zeros(args.num_bins[1])
        meanPHb0 = np.zeros(args.num_bins[2])
        meanPHb1 = np.zeros(args.num_bins[3])
        ls0 = np.linspace(0,args.max_life,args.num_bins[0])
        ls1 = np.linspace(0,args.max_life,args.num_bins[1])
        for fname in tqdm(fns, total=len(fns)):
            bfn = os.path.splitext(os.path.basename(fname))[0]
            if args.imgtype=="npy":
                ph = np.load(fname)
            else:
                sample = np.array(Image.open(fname).convert('L'),dtype=np.float64)
                ph = comp_PH(sample)
                np.save(os.path.join(PHdir, bfn), ph)
            hs = hist_PH(ph, ls0, ls1, args.num_bins, args.max_life, args.max_birth, args.bandwidth)
            meanPHl0 += hs[:args.num_bins[0]]
            meanPHl1 += hs[args.num_bins[0]:(args.num_bins[0]+args.num_bins[1])]
            meanPHb0 += hs[(args.num_bins[0]+args.num_bins[1]):(args.num_bins[0]+args.num_bins[1]+args.num_bins[2])]
            meanPHb1 += hs[(args.num_bins[0]+args.num_bins[1]+args.num_bins[2]):]
            np.save(os.path.join(PHdir, bfn+"hist"), hs.astype(np.float32))

        meanPHl0 /= len(fns)
        meanPHl1 /= len(fns)
        meanPHb0 /= len(fns)
        meanPHb1 /= len(fns)
        print(meanPHl0.max(), meanPHl1.max(),meanPHb0.max(),meanPHb1.max())
        sns.lineplot(np.arange(len(meanPHl0)),meanPHl0, legend="full")
        sns.lineplot(np.arange(len(meanPHl1)),meanPHl1, legend="full")
        sns.lineplot(np.arange(len(meanPHb0)),meanPHb0, legend="full")
        sns.lineplot(np.arange(len(meanPHb1)),meanPHb1, legend="full")
        plt.show()
    elif args.compute_persistence_image:
        import persim
        task = partial(comp_save_persistence_image, args=args)
        pool = Pool(args.num_workers)
        with tqdm(total=len(fns)) as t:
            for _ in pool.imap_unordered(task, fns):
                t.update(1)

    else:
        task = partial(comp_save_PH, PHdir=PHdir)
        pool = Pool(args.num_workers)
        with tqdm(total=len(fns)) as t:
            for _ in pool.imap_unordered(task, fns):
                t.update(1)
