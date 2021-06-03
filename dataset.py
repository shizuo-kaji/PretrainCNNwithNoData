# -*- coding: utf-8 -*-
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset

from PIL import Image

import os,glob
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from PHdict import comp_PH, hist_PH, life_curve, comp_persistence_image
from scipy.fft import fft2, ifft2
from skimage.filters import threshold_otsu


def generate_random_image(args):
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
    sample = Image.fromarray((255*sample).astype(np.uint8).transpose(1,2,0).squeeze())
    return sample
    

class DatasetFolderPH(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = None,
            transform: Optional[Callable] = None,
            args = None
    ) -> None:
        super(DatasetFolderPH, self).__init__(root, transform=transform)

        self.args = args
        if root is None:
            self.generate_on_the_fly = True
            self.n_samples = args.n_samples
        else:
            self.generate_on_the_fly = False
            self.samples = glob.glob(os.path.join(self.root, "**/*.png"), recursive=True)
            self.samples.extend(glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True))
            self.n_samples = len(self.samples)

        if self.n_samples == 0:
            msg = "no files found in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if loader is None:
            self.loader = default_loader
        else:
            self.loader = loader

        bins = args.numof_classes
        if self.args.label_type in ["life_curve","persistence_image"]:
            self.args.num_bins = [bins//2,bins-bins//2]
        elif self.args.label_type == "PH_hist":
            b = bins//4
            self.args.num_bins = [b,b,b,bins-3*b]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if self.generate_on_the_fly:
            sample=generate_random_image(self).convert('RGB')
        else:            
            path = self.samples[index]
            sample = self.loader(path)

        if self.args.persistence_after_transform and self.transform is not None:
            sample = self.transform(sample)

        if self.args.label_type == "raw":
            hs = np.load(os.path.join(self.args.path2PHdir, os.path.splitext(os.path.basename(path))[0]+".npy")).astype(np.float32)
        else:
            if self.args.path2PHdir:
                ph = np.load(os.path.join(self.args.path2PHdir, os.path.splitext(os.path.basename(path))[0]+".npy"))
            else: # compute on the fly
                ph = comp_PH(np.array(sample),gradient=self.args.gradient, distance_transform=self.args.distance_transform)
            if self.args.label_type == "life_curve":
                hs = life_curve(ph, self.args.num_bins, min_birth=self.args.min_birth, max_birth=self.args.max_birth, max_life=self.args.max_life).astype(np.float32)
            elif self.args.label_type == "PH_hist":
                hs = hist_PH(ph, self.args).astype(np.float32)
            elif self.args.label_type == "persistence_image":
                hs = comp_persistence_image(ph, self.args).astype(np.float32)       
            else:
                print("Unknown label type")
                exit()
    #        print(hs.shape, hs.max())

        if not self.args.persistence_after_transform:
            if self.transform is not None:
                sample = self.transform(sample)
        return sample, hs

    def __len__(self) -> int:
        return self.n_samples


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
