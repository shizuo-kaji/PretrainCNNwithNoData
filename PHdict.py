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
from skimage.filters import threshold_otsu, scharr
from PIL import Image
from multiprocessing import Pool
from functools import partial
try:
    import persim
except:
    pass

def preprocess_image(img, gradient=False, distance_transform=True, binarize=False):
    if len(img.shape)>2:
        im = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        im = img
    if gradient:
        im = scharr(im)
    if distance_transform:
        if im.max()==im.min():
            return(np.zeros_like(im))
        bw_img = (im >= threshold_otsu(im))
        if gradient:
            dt_img = distance_transform_edt(~bw_img)
        else:
            dt_img = distance_transform_edt(bw_img)-distance_transform_edt(~bw_img)
        return(dt_img)
    elif binarize:
        return(im >= threshold_otsu(im))
    else:
        return(im)

def comp_PH(img, distance_transform=True, gradient=True):
    im = preprocess_image(img, gradient=gradient, distance_transform=distance_transform)
    pd = cripser.computePH(im)
    #pd = cripser.computePH(im,maxdim=args.maxdim,top_dim=args.top_dim,embedded=args.embedded)
    #print(sum(pd[:,0] == 1))
    #print(im.shape)
    return(pd)

def comp_save_PH(fname, args):
    bfn = os.path.splitext(os.path.basename(fname))[0]
    img = np.array(Image.open(fname).convert('L'),dtype=np.float64)
    ph = comp_PH(img, gradient=args.gradient, distance_transform=args.distance_transform)
    np.save(os.path.join(args.output, bfn), ph)

def comp_save_persistence_image(fname, args=None):
    bfn = os.path.splitext(os.path.basename(fname))[0]
    img = np.array(Image.open(fname).convert('L'),dtype=np.float64)
    ph = comp_PH(img, gradient=args.gradient, distance_transform=args.distance_transform)
    pims = comp_persistence_image(ph, args)
    np.save(os.path.join(args.output, bfn+"_persImg"), np.concatenate(pims).astype(np.float32))
    if args.save_fig:
        sns.lineplot(x=np.arange(len(pims[0])),y=pims[0], legend="full")
        sns.lineplot(x=np.arange(len(pims[1])),y=pims[1], legend="full",style=True, dashes=[(2,2)])
        plt.savefig(os.path.join(args.output, bfn+"_persImg.jpg"))
        plt.close()
    return(pims)

def comp_persistence_image(ph, args=None):
    pims = []
    for d in [0,1]:
        s = np.sqrt((args.max_birth-args.min_birth)*args.max_life/args.num_bins[d])
        #print(s, args.num_bins[d])
        pim = persim.PersistenceImager(birth_range=(args.min_birth,args.max_birth), pers_range=(0,args.max_life),pixel_size=s,kernel_params={'sigma': [[args.persImg_sigma, 0.0], [0.0, args.persImg_sigma]]})
        p = (ph[ph[:,0]==d])[:,1:3]
        life = p[:,1]-p[:,0]
        life = np.clip(life,a_min=None,a_max=args.max_life)
        p[:,1] = life
        pims.append(np.sqrt(np.abs(pim.transform(p).ravel())).astype(np.float32))
    return(np.concatenate(pims))

def life_curve(ph, dim, min_birth=None, max_birth=None, max_life=None):
    res = []
    for d in range(2):   
        pds =ph[ph[:,0] == d, 1:3]
        mlife = (np.clip(pds[:,1]-pds[:,0],0,max_life))

        res.append(np.zeros(dim[d]))
        for i,th in enumerate(np.linspace(min_birth,max_birth,num=dim[d])):
            #print(th, np.sum(np.logical_and(pds[:,0] < th, pds[:,1] > th)))
            res[-1][i] = np.sum(mlife[np.logical_and(pds[:,0] < th, pds[:,1] > th)])

    return np.sqrt(np.concatenate(res))


def hist_PH(ph, args):
    pds =[ph[ph[:,0] == i, 1:3] for i in range(2)]
    #print(len(pds[0]),len(pds[1]))
    life = [pd[:,1]-pd[:,0] for pd in pds]
    life = [l[l<1e+10] for l in life]  # remove permanent cycle
    life = [np.clip(l,0,args.max_life) for l in life]
    # hsb = np.zeros(args.num_bins[2])
    # for k,ind in enumerate(np.searchsorted(lsb,pds[1][:,0])):
    #     hsb[ind] += (pds[1][k,1]-pds[1][k,0]) ## lifetime weighted count
    # hs0 = gaussian_kde(life[0])(ls) * len(life[0])
    # hs1 = gaussian_kde(life[1])(ls) * len(life[1])
    hsl0, _ = np.histogram(life[0],bins=args.num_bins[0], range=(0,args.max_life))
    hsl1, _ = np.histogram(life[1],bins=args.num_bins[1], range=(0,args.max_life))
    # plt.hist(pds[0][:,0],weights=pds[0][:,1]-pds[0][:,0])
    # plt.show()
    # plt.hist(pds[1][:,0],weights=pds[1][:,1]-pds[1][:,0])
    # plt.show()
    max_birth_for_h0 = args.max_birth if args.gradient else 0 ## all 0-cycles are born before 0
    hsb0, _ = np.histogram(pds[0][:,0],bins=args.num_bins[2], range=(args.min_birth,max_birth_for_h0), weights=pds[0][:,1]-pds[0][:,0]) 
    hsb1, _ = np.histogram(pds[1][:,0],bins=args.num_bins[3], range=(args.min_birth,args.max_birth), weights=pds[1][:,1]-pds[1][:,0])
    # lifetime weighting
    hsl0 = hsl0*(1.0+np.linspace(0,args.max_life,args.num_bins[0]))
    hsl1 = hsl1*(1.0+np.linspace(0,args.max_life,args.num_bins[1]))
    # smoothing
    hsl0 = kern_smooth(hsl0, bandwidth=args.bandwidth, kern='hanning')
    hsl1 = kern_smooth(hsl1, bandwidth=args.bandwidth, kern='hanning')
    hsb0 = kern_smooth(hsb0, bandwidth=args.bandwidth, kern='hanning')
    hsb1 = kern_smooth(hsb1, bandwidth=args.bandwidth, kern='hanning')
    # log
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
    parser.add_argument('--max_birth', '-maxb', type=int, default=50)
    parser.add_argument('--min_birth', '-minb', type=int, default=None)
    parser.add_argument('--num_bins', '-n', type=int, nargs="*", default=[50,50,50,50])
    parser.add_argument('--bandwidth', '-b', type=int, default=1)
    parser.add_argument('--persImg_sigma', '-s', type=float, default=100)
    parser.add_argument('--imgtype', '-it', type=str, default="jpg")
    parser.add_argument('--type', '-t', type=str, help="type of label")
    parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers (data_loader)")
    parser.add_argument('--save_fig', '-sf', action="store_true", help="save graphs")
    parser.add_argument('--distance_transform', '-dt', action="store_true", default=True, help="apply distance transform")
    parser.add_argument('--gradient', '-g', action="store_true", default=False, help="apply gradient filter")

    args = parser.parse_args()
    if args.min_birth is None:
        if args.gradient or not args.distance_transform:
            args.min_birth = 0
        else:
            args.min_birth = -args.max_birth

    target_dir = args.target_dir
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    fns=sorted(glob.glob(os.path.join(target_dir,"**/*.{}".format(args.imgtype)), recursive=True))

    if args.type == "PH_hist":
        print("compute and save persistence histogram...")
        meanPHl0 = np.zeros(args.num_bins[0])
        meanPHl1 = np.zeros(args.num_bins[1])
        meanPHb0 = np.zeros(args.num_bins[2])
        meanPHb1 = np.zeros(args.num_bins[3])
        for fname in tqdm(fns, total=len(fns)):
            bfn = os.path.splitext(os.path.basename(fname))[0]
            if args.imgtype=="npy":
                ph = np.load(fname)
            else:
                sample = np.array(Image.open(fname).convert('L'),dtype=np.float64)
                ph = comp_PH(sample, gradient=args.gradient, distance_transform=args.distance_transform)
                np.save(os.path.join(args.output, bfn), ph)
            hs = hist_PH(ph, args)
            np.save(os.path.join(args.output, bfn+"_hist"), hs.astype(np.float32))
            c1 = hs[:args.num_bins[0]]
            c2 = hs[args.num_bins[0]:(args.num_bins[0]+args.num_bins[1])]
            c3 = hs[(args.num_bins[0]+args.num_bins[1]):(args.num_bins[0]+args.num_bins[1]+args.num_bins[2])]
            c4 = hs[(args.num_bins[0]+args.num_bins[1]+args.num_bins[2]):]
            meanPHl0 += c1
            meanPHl1 += c2
            meanPHb0 += c3
            meanPHb1 += c4
            if args.save_fig:
                sns.lineplot(x=np.arange(len(c1)),y=c1, legend="full")
                sns.lineplot(x=np.arange(len(c2)),y=c2, legend="full",style=True, dashes=[(2,2)])
                sns.lineplot(x=np.arange(len(c3)),y=c3, legend="full",linewidth=2.5)
                sns.lineplot(x=np.arange(len(c4)),y=c4, legend="full",style=True, dashes=[(2,2)],linewidth=2.5)
                plt.savefig(os.path.join(args.output, bfn+"_histCurve.jpg"))
                plt.close()

        meanPHl0 /= len(fns)
        meanPHl1 /= len(fns)
        meanPHb0 /= len(fns)
        meanPHb1 /= len(fns)
        print(meanPHl0.max(), meanPHl1.max(),meanPHb0.max(),meanPHb1.max())
        if args.save_fig:
            sns.lineplot(x=np.arange(len(meanPHl0)),y=meanPHl0, legend="full")
            sns.lineplot(x=np.arange(len(meanPHl1)),y=meanPHl1, legend="full",style=True, dashes=[(2,2)])
            sns.lineplot(x=np.arange(len(meanPHb0)),y=meanPHb0, legend="full",linewidth=2.5)
            sns.lineplot(x=np.arange(len(meanPHb1)),y=meanPHb1, legend="full",style=True, dashes=[(2,2)],linewidth=2.5)
            plt.show()
    elif args.type == "life_curve":
        print("compute and save life curve...")
        meanPHl0 = np.zeros(args.num_bins[0])
        meanPHl1 = np.zeros(args.num_bins[1])
        for fname in tqdm(fns, total=len(fns)):
            sample = np.array(Image.open(fname).convert('L'),dtype=np.float64)
            ph = comp_PH(sample, gradient=args.gradient, distance_transform=args.distance_transform)
            res = life_curve(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life)
            bfn = os.path.splitext(os.path.basename(fname))[0]
            np.save(os.path.join(args.output, bfn+"_lifeCurve"), res.astype(np.float32))
            c1 = res[:args.num_bins[0]]
            c2 = res[args.num_bins[0]:]
            if args.save_fig:
                sns.lineplot(x=np.arange(len(c1)),y=c1, legend="full")
                sns.lineplot(x=np.arange(len(c2)),y=c2, legend="full",style=True, dashes=[(2,2)])
                plt.savefig(os.path.join(args.output, bfn+"_lifeCurve.jpg"))
                plt.close()
            meanPHl0 += c1
            meanPHl1 += c2
        meanPHl0 /= len(fns)
        meanPHl1 /= len(fns)
        print(meanPHl0.max(), meanPHl1.max())
        if args.save_fig:
            sns.lineplot(x=np.arange(len(meanPHl0)),y=meanPHl0, legend="full")
            sns.lineplot(x=np.arange(len(meanPHl1)),y=meanPHl1, legend="full",style=True, dashes=[(2,2)])
            plt.show()
    elif args.type == "persistence_image":
        print("compute and save persistence images...")
        task = partial(comp_save_persistence_image, args=args)
        with Pool(args.num_workers) as pool:
            with tqdm(total=len(fns)) as t:
                for _ in pool.imap_unordered(task, fns):
                    t.update(1)
    elif args.type == "grid":
        fig = plt.figure(figsize=(21,10),tight_layout=True)
        n = min(len(fns),10)
        axes = fig.subplots(n, 6)
        for i in tqdm(range(n)):
            fname = fns[i]
            colour = Image.open(fname)
            sample = (np.array(colour.convert('L'),dtype=np.float64))
            mask = preprocess_image(sample, gradient=args.gradient, distance_transform=False, binarize=True)
            dt = preprocess_image(sample, gradient=args.gradient, distance_transform=args.distance_transform, binarize=True)
            print(dt.min(),dt.max())
            ph = comp_PH(sample, gradient=args.gradient, distance_transform=args.distance_transform)
            axes[i,0].imshow(colour)
            axes[i,0].set_title(os.path.basename(fns[i]))
            axes[i,1].imshow(mask)
            axes[i,2].imshow(dt,vmin=args.min_birth,vmax=args.max_birth)
            res = life_curve(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life)
            sns.lineplot(x=np.arange(args.num_bins[0]),y=res[:args.num_bins[0]], ax=axes[i,3])
            sns.lineplot(x=np.arange(args.num_bins[1]),y=res[args.num_bins[0]:], style=True, dashes=[(2,3)], ax=axes[i,3])
            res = hist_PH(ph, args)
            sns.lineplot(x=np.arange(args.num_bins[0]),y=res[:args.num_bins[0]], ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[1]),y=res[args.num_bins[0]:(args.num_bins[0]+args.num_bins[1])], style=True, dashes=[(2,2)], ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[2]),y=res[(args.num_bins[0]+args.num_bins[1]):(args.num_bins[0]+args.num_bins[1]+args.num_bins[2])],linewidth=2.5, ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[3]),y=res[(args.num_bins[0]+args.num_bins[1]+args.num_bins[2]):], style=True, dashes=[(2,2)],linewidth=2.5, ax=axes[i,4])
            res = comp_persistence_image(ph, args)
            sns.lineplot(x=np.arange(args.num_bins[0]),y=res[:args.num_bins[0]], ax=axes[i,5])
            sns.lineplot(x=np.arange(len(res[args.num_bins[0]:])),y=res[args.num_bins[0]:], style=True, dashes=[(2,2)], ax=axes[i,5])
            for ax in axes[i]:
                ax.legend([],[], frameon=False)
        plt.savefig("persistence_vectors.jpg")
        plt.show()

    ## compute persistence diagrams only
    else:
        print("compute and save persistent homology...")
        task = partial(comp_save_PH, args=args)
        with Pool(args.num_workers) as pool:
            with tqdm(total=len(fns)) as t:
                for _ in pool.imap_unordered(task, fns):
                    t.update(1)
