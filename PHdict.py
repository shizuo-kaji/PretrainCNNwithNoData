# -*- coding: utf-8 -*-
# Computing Persistent Homology and its histogram
## install cripser by
## pip install git+https://github.com/shizuo-kaji/CubicalRipser_3dim

import os,glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import argparse,json
from scipy.ndimage.morphology import distance_transform_edt
from skimage import feature,morphology
from skimage.filters import threshold_otsu, threshold_niblack,threshold_sauvola, scharr
from PIL import Image
from multiprocessing import Pool
from functools import partial
from arguments import arguments
try:
    import persim
    import cripser
    from gudhi.representations import Landscape
except:
    pass

# preprocess image before computing PH
def preprocess_image(img, gradient=False, img_size=None, filtration=None, origin=(0,0), binarisation='otsu'):
    im = img

    if img_size:
        import skimage.transform
        im = skimage.transform.resize(im,(img_size,img_size))

    if gradient:
        import cv2
        #im = feature.canny(im, sigma=10)
        #im = scharr(im)
        if len(im.shape)==2: # temporaly convert to BGR to use cv2 functions
            im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(im, None, 30, 10, 7, 21)
        im = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        v = np.median(im)
        sigma = 0.9 # 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        im = cv2.Canny(im, lower, upper)
    
    if len(im.shape)>2:
        im = np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])

    if filtration is not None:
        if im.max()==im.min():
            return(np.zeros_like(im))
        if gradient:
            bw_img = im
        else:
            if binarisation=='sauvola':
                bw_img = (im >=threshold_sauvola(im))
            elif binarisation=='niblack':
                bw_img = (im >= threshold_niblack(im))            
            else:           
                bw_img = (im >= threshold_otsu(im))
        #bw_img = morphology.area_opening(bw_img, area_threshold=8, connectivity=2)

        if filtration=='binarise':
            return(bw_img)
        elif filtration == 'distance':
            dt_img = distance_transform_edt(~bw_img) # distance from the foreground 0
            #print(dt_img.max())
        elif filtration == 'signed_distance':
            dt_img = distance_transform_edt(~bw_img)-distance_transform_edt(bw_img)
        elif filtration in ['downward','upward']:
            null_idx = (bw_img == 0)
            ## height transform
            if len(im.shape) == 3: #(z,y,x)
                h = np.arange(im.shape[0]).reshape(-1,1,1)
            else:
                h = np.arange(im.shape[0]).reshape(-1,1)
            if filtration=='upward':
                h = np.max(h) - h
            dt_img = (bw_img * h)
            dt_img[null_idx] = np.max(h)
        elif 'radial' in filtration:
            null_idx = (bw_img == 0)
            h = np.linalg.norm(np.stack(np.meshgrid(*map(range,im.shape),indexing='ij'),axis=-1)-np.array(origin), axis=-1)
            dt_img = (bw_img * h)
            if filtration=='radial_inv':
                dt_img = np.max(dt_img) - dt_img
            else:
                dt_img[null_idx] = np.max(h)

        dt_img *= 256/dt_img.shape[0]  # scaling normalisation
        return(dt_img)
    else:
        return(im)

## computing PH of an image
def comp_PH(img, gradient=True, img_size=None, filtration=None):
    im = preprocess_image(img, gradient=gradient, img_size=img_size, filtration=filtration)
    pd = cripser.computePH(im.astype(np.float64))
    #pd = cripser.computePH(im,maxdim=args.maxdim,top_dim=args.top_dim,embedded=args.embedded)
    #print(sum(pd[:,0] == 1))
    #print(im.shape)
    return(pd)

def comp_save_PH(fname, args):
    bfn = os.path.splitext(os.path.basename(fname))[0]
    img = np.array(Image.open(fname).convert('L'),dtype=np.float64)
    ph = comp_PH(img, gradient=args.gradient, filtration=args.filtration, img_size=args.img_size)
    np.save(os.path.join(args.output, bfn), ph)

def comp_landscape(ph,  dim, min_birth=None, max_birth=None, max_life=None,n=5):
    res = []
    for d in [0,1]:
        pds = ph[ph[:,0] == d, 1:3]
        #pds[:,1] = pds[:,0]+(np.clip(pds[:,1]-pds[:,0],0,max_life))
        res.append(Landscape(num_landscapes=n, resolution=dim[d]//n, sample_range=[min_birth[d],max_birth[d]]).fit_transform([pds]).ravel().astype(np.float32))
    return(np.sqrt(np.concatenate(res)))

def comp_persistence_image(ph, args=None):
    pims = []
    for d in [0,1]:
        s = np.sqrt((args.max_birth[d]-args.min_birth[d])*args.max_life[d]/args.num_bins[d])
        p = int((args.max_birth[d]-args.min_birth[d])/s)
        q = int(args.max_life[d]/s)
        while p*q < args.num_bins[d]:
            s = max((args.max_birth[d]-args.min_birth[d])/(p+1), args.max_life[d]/(q+1))
            p = int((args.max_birth[d]-args.min_birth[d])/s)
            q = int(args.max_life[d]/s)
        pim = persim.PersistenceImager(birth_range=(args.min_birth[d],args.max_birth[d]),
            pers_range=(0,args.max_life[d]),pixel_size=s,
            kernel_params={'sigma': [[args.persImg_sigma, 0.0], [0.0, args.persImg_sigma]]},
            weight_params={'n': args.persImg_weight})
        p = (ph[ph[:,0]==d])[:,1:3] # extract dim=d cycles
        life = p[:,1]-p[:,0]
        life = np.clip(life,a_min=None,a_max=args.max_life[d])
        p[:,1] = life
        pi = pim.transform(p, skew=False)
        #print(s, args.num_bins[d])
        #print(pi.shape)
        pi = pi.ravel()
        #pi = np.pad(pi,(0,args.num_bins[d]))
        pi = pi[:args.num_bins[d]]
        pi = np.abs(pi) ** args.persImg_power # to suppress overflow during learning
        pims.append(pi.astype(np.float32))
    return(pims)

def comp_betticurve(ph, dim, min_birth=None, max_birth=None, max_life=None):
    res = []
    for d in range(2):
        pds = ph[ph[:,0] == d, 1:3]
        mlife = (np.clip(pds[:,1]-pds[:,0],0,max_life[d]))

        res.append(np.zeros(dim[d]))
        for i,th in enumerate(np.linspace(min_birth[d],max_birth[d],num=dim[d])):
            #print(th, np.sum(np.logical_and(pds[:,0] < th, pds[:,1] > th)))
            res[-1][i] = np.sum(mlife[np.logical_and(pds[:,0] < th, pds[:,1] > th)])

    return np.sqrt(np.concatenate(res))


def comp_persistence_histogram(ph, num_bins, min_birth=None, max_birth=None, max_life=None, bandwidth=1):
#    print(args.num_bins)
    pds =[ph[ph[:,0] == i, 1:3] for i in range(2)]
    #print(len(pds[0]),len(pds[1]))
    life = [pd[:,1]-pd[:,0] for pd in pds]
    life = [l[l<1e+10] for l in life]  # remove permanent cycle
    life = [np.clip(l,0,max_life[i]) for i,l in enumerate(life)]
    # hsb = np.zeros(args.num_bins[2])
    # for k,ind in enumerate(np.searchsorted(lsb,pds[1][:,0])):
    #     hsb[ind] += (pds[1][k,1]-pds[1][k,0]) ## lifetime weighted count
    # hs0 = gaussian_kde(life[0])(ls) * len(life[0])
    # hs1 = gaussian_kde(life[1])(ls) * len(life[1])
    ## histogram for lifetime for each dimension
    hsl0, _ = np.histogram(life[0],bins=num_bins[0], range=(0,max_life[0]))
    hsl1, _ = np.histogram(life[1],bins=num_bins[1], range=(0,max_life[1]))
    # plt.hist(pds[0][:,0],weights=pds[0][:,1]-pds[0][:,0])
    # plt.show()
    # plt.hist(pds[1][:,0],weights=pds[1][:,1]-pds[1][:,0])
    # plt.show()
    ## histogram for birthtime for each dimension
    hsb0, _ = np.histogram(pds[0][:,0],bins=num_bins[2], range=(min_birth[0],max_birth[0]), weights=pds[0][:,1]-pds[0][:,0])
    hsb1, _ = np.histogram(pds[1][:,0],bins=num_bins[3], range=(min_birth[1],max_birth[1]), weights=pds[1][:,1]-pds[1][:,0])
    # lifetime weighting
    hsl0 = hsl0*(1.0+np.linspace(0,max_life[0],num_bins[0]))
    hsl1 = hsl1*(1.0+np.linspace(0,max_life[1],num_bins[1]))
    # smoothing
    hsl0 = kern_smooth(hsl0, bandwidth=bandwidth, kern='hanning')
    hsl1 = kern_smooth(hsl1, bandwidth=bandwidth, kern='hanning')
    hsb0 = kern_smooth(hsb0, bandwidth=bandwidth, kern='hanning')
    hsb1 = kern_smooth(hsb1, bandwidth=bandwidth, kern='hanning')
    # log and scaling
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

    args = arguments(mode="PHdict")

    grad = "grad" if args.gradient else ""
    if args.output is None:
        dn1,dn2 = os.path.split((os.path.normpath(args.target_dir))) # the leaf name
        phdn = os.path.join(dn1,"PH{}_{}_{}".format(grad,args.filtration,dn2))
        # if os.path.isdir(phdn):
        #     print("Please specify output directory!")
        #     exit()
        # else:
        args.output = phdn
        print("output will be saved under: ", args.output)

    ###
    print(args)
    target_dir = args.target_dir
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    gfns = []
    imgtypes = ['png','PNG','jpg','JPG','tif','TIF','tiff','TIFF']
    for it in imgtypes:
        gfns.extend(glob.glob(os.path.join(target_dir,"**/*.{}".format(it)), recursive=True))
    fns=sorted(list(set(gfns)))

    if args.label_type_pt == "persistence_histogram":
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
                ph = comp_PH(sample, gradient=args.gradient, filtration=args.filtration)
                np.save(os.path.join(args.output, bfn), ph)
            hs = comp_persistence_histogram(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life, bandwidth=args.bandwidth)
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
        print(sum(meanPHl0>0), sum(meanPHl1>0), sum(meanPHb0>0), sum(meanPHb1>0))
        if args.save_fig:
            sns.lineplot(x=np.arange(len(meanPHl0)),y=meanPHl0, legend="full")
            sns.lineplot(x=np.arange(len(meanPHl1)),y=meanPHl1, legend="full",style=True, dashes=[(2,2)])
            sns.lineplot(x=np.arange(len(meanPHb0)),y=meanPHb0, legend="full",linewidth=2.5)
            sns.lineplot(x=np.arange(len(meanPHb1)),y=meanPHb1, legend="full",style=True, dashes=[(2,2)],linewidth=2.5)
            plt.show()
    elif args.label_type_pt == "persistence_betticurve":
        print("compute and save betti curve...")
        meanPHl0 = np.zeros(args.num_bins[0])
        meanPHl1 = np.zeros(args.num_bins[1])
        for fname in tqdm(fns, total=len(fns)):
            sample = np.array(Image.open(fname).convert('L'),dtype=np.float64)
            ph = comp_PH(sample, gradient=args.gradient, filtration=args.filtration)
            res = comp_betticurve(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life)
            bfn = os.path.splitext(os.path.basename(fname))[0]
            np.save(os.path.join(args.output, bfn+"_bettiCurve"), res.astype(np.float32))
            c1 = res[:args.num_bins[0]]
            c2 = res[args.num_bins[0]:]
            if args.save_fig:
                sns.lineplot(x=np.arange(len(c1)),y=c1, legend="full")
                sns.lineplot(x=np.arange(len(c2)),y=c2, legend="full",style=True, dashes=[(2,2)])
                plt.savefig(os.path.join(args.output, bfn+"_bettiCurve.jpg"))
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
    elif args.label_type_pt == "persistence_image":
        print("compute and save persistence images...")
        for fname in tqdm(fns, total=len(fns)):
            bfn = os.path.splitext(os.path.basename(fname))[0]
            img = Image.open(fname).convert('L')
            img = np.array(img, dtype=np.float64)
            ph = comp_PH(img, gradient=args.gradient, img_size=args.img_size, filtration=args.filtration)
            pims = comp_persistence_image(ph, args)
            np.save(os.path.join(args.output, bfn+"_persImg"), np.concatenate(pims).astype(np.float32))
            if args.save_fig:
                sns.lineplot(x=np.arange(len(pims[0])),y=pims[0], legend="full")
                sns.lineplot(x=np.arange(len(pims[1])),y=pims[1], legend="full",style=True, dashes=[(2,2)])
                plt.savefig(os.path.join(args.output, bfn+"_persImg.jpg"))
                plt.close()
                # plt.imshow(pims[0].reshape(10,5))
                # plt.savefig(os.path.join(args.output, bfn+"_persImg0.jpg"))
                # plt.close()
                # plt.imshow(pims[1].reshape(10,5))
                # plt.savefig(os.path.join(args.output, bfn+"_persImg1.jpg"))
                # plt.close()
    elif args.label_type_pt == "grid":
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize=(21,10),tight_layout=True)
        n = min(len(fns),6)
        axes = fig.subplots(n, 6)
        for i in tqdm(range(n)):
            fname = fns[i]
            sample = np.array(Image.open(fname))
            mask = preprocess_image(sample, gradient=args.gradient, filtration='binarise', img_size=args.img_size) ## used only for preview
            dt = preprocess_image(sample, gradient=args.gradient, filtration=args.filtration, img_size=args.img_size)
            print(dt.min(),dt.max())
            ph = comp_PH(sample, gradient=args.gradient, filtration=args.filtration, img_size=args.img_size)
            axes[i,0].imshow(sample)
            axes[i,0].set_axis_off()
            axes[i,0].set_title(os.path.basename(fns[i]))
            axes[i,1].imshow(mask)
            axes[i,1].set_axis_off()
            im2 = axes[i,2].imshow(dt,vmin=args.min_birth[0],vmax=args.max_birth[0], )
            axes[i,2].set_axis_off()
            divider = make_axes_locatable(axes[i,2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax, orientation='vertical')

            axes[i,3].set_title("LC")
            res = comp_betticurve(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life)
            #print(ph)
            sns.lineplot(x=np.arange(args.num_bins[0]),y=res[:args.num_bins[0]], ax=axes[i,3])
            sns.lineplot(x=np.arange(args.num_bins[1]),y=res[args.num_bins[0]:], style=True, dashes=[(2,3)], ax=axes[i,3])
            axes[i,4].set_title("HS")
            res = comp_persistence_histogram(ph, args.num_bins, min_birth=args.min_birth, max_birth=args.max_birth, max_life=args.max_life)
            sns.lineplot(x=np.arange(args.num_bins[0]),y=res[:args.num_bins[0]], ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[1]),y=res[args.num_bins[0]:(args.num_bins[0]+args.num_bins[1])], style=True, dashes=[(2,2)], ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[2]),y=res[(args.num_bins[0]+args.num_bins[1]):(args.num_bins[0]+args.num_bins[1]+args.num_bins[2])],linewidth=2.5, ax=axes[i,4])
            sns.lineplot(x=np.arange(args.num_bins[3]),y=res[(args.num_bins[0]+args.num_bins[1]+args.num_bins[2]):], style=True, dashes=[(2,2)],linewidth=2.5, ax=axes[i,4])
            axes[i,5].set_title("PI")
            res = comp_persistence_image(ph, args)
            #print(res[0].shape)
            sns.lineplot(x=np.arange(len(res[0])),y=res[0], ax=axes[i,5])
            sns.lineplot(x=np.arange(len(res[0])),y=res[1], style=True, dashes=[(2,2)], ax=axes[i,5])
            for ax in axes[i]:
                ax.legend([],[], frameon=False)
        plt.savefig(os.path.join(args.output,"persistence_vectors.jpg"))
        plt.show()

    ## compute persistence diagrams only
    else:
        print("compute and save persistent homology...")
        task = partial(comp_save_PH, args=args)
        with Pool(args.num_workers) as pool:
            with tqdm(total=len(fns), ascii=True, ncols=100) as t:
                for _ in pool.imap_unordered(task, fns):
                    t.update(1)
