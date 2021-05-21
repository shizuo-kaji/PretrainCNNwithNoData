# -*- coding: utf-8 -*-
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset

from PIL import Image

import os,glob
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
from PHdict import comp_PH, hist_PH
from scipy.fft import fft2, ifft2
from skimage.filters import threshold_otsu

class DatasetFolderPH(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = None,
            transform: Optional[Callable] = None,
            persistence = None, bins = 200, max_life=50, PHdir=None, precomputed=False,
    ) -> None:
        super(DatasetFolderPH, self).__init__(root, transform=transform)

        self.samples = glob.glob(os.path.join(self.root, "**/*.png"), recursive=True)
        self.samples.extend(glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True))

        if len(self.samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if loader is None:
            self.loader = default_loader
        else:
            self.loader = loader
        self.max_life = max_life
        self.bins = bins
        self.PHdir = PHdir
        self.persistence = persistence
        if PHdir and persistence != "pre":
            print("precomputation of PH is only valid for pre-transformation!")
            exit()
        self.max_life = max_life
        self.max_birth = max_life
        self.precomputed = precomputed
        self.num_bins = [bins//4,bins//4,bins//4,bins//4]
        self.ls0 = np.linspace(0,max_life,self.num_bins[0])
        self.ls1 = np.linspace(0,max_life,self.num_bins[1])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.samples[index]
        sample = self.loader(path)

        if self.persistence != 'pre':
            if self.transform is not None:
                sample = self.transform(sample)

        if self.precomputed:
            hs = np.load(os.path.join(self.PHdir, os.path.splitext(os.path.basename(path))[0]+".npy")).astype(np.float32)
        else:
            if self.PHdir:
                ph = np.load(os.path.join(self.PHdir, os.path.splitext(os.path.basename(path))[0]+".npy"))
            else: # compute on the fly
                ph = comp_PH(np.array(sample))
            hs = hist_PH(ph, self.ls0, self.ls1, self.num_bins, self.max_life, self.max_birth, bandwidth=1).astype(np.float32)
    #        print(hs.shape, hs.max())

        if self.persistence == 'pre':
            if self.transform is not None:
                sample = self.transform(sample)
        return sample, hs

    def __len__(self) -> int:
        return len(self.samples)


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


## generate data on the fly (not efficient)
class DatasetGeneratePH(Dataset):
    def __init__(
            self, n_samples=10000, size=256,
            alpha_range = (0.01,1),
            beta_range = (0.5,2),
            prob_binary = 1,
            transform: Optional[Callable] = None,
            persistence = None, bins = 200, max_life=50, PHdir=None,
    ) -> None:
        self.n_samples = n_samples
        self.transform = transform
        self.size = size
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.max_life = max_life
        self.bins = bins
        self.max_life = max_life
        self.max_birth = max_life
        self.num_bins = [bins//4,bins//4,bins//4,bins//4]
        self.ls0 = np.linspace(0,max_life,self.num_bins[0])
        self.ls1 = np.linspace(0,max_life,self.num_bins[1])
        self.prob_binary = prob_binary
        self.persistence = persistence

    def __getitem__(self, index: int):
#        print(alpha,beta)
        sample=np.zeros(1)
        while sample.max()-sample.min()<1e-10:
            alpha = np.random.uniform(*self.alpha_range)
            beta = np.random.uniform(*self.beta_range)
            x = np.linspace(1,np.exp(alpha)*self.size,self.size)
            X, Y = np.meshgrid(x,x)
            noise = np.random.uniform(0,1,(self.size,self.size))
            f = fft2(noise)
            f = f/(X**2+Y**2)**beta
            sample = ifft2(f).real
        #print(sample.min(), sample.max())
        p = np.random.uniform(0,1)
        if p<self.prob_binary/2:
            sample = (sample >= threshold_otsu(sample))
        elif p<self.prob_binary:
            sample = (sample < threshold_otsu(sample))
        else:
            sample = (sample-sample.min())/np.ptp(sample)

        sample = np.repeat(sample[np.newaxis,:],3,axis=0).astype(np.float32)
        # does not work now:
        if self.persistence != 'pre':
            if self.transform is not None:
                sample = self.transform(Image.fromarray((255*sample).astype(np.uint8).transpose(1,2,0)))

        ph = comp_PH(sample[0])
        hs = hist_PH(ph, self.ls0, self.ls1, self.num_bins, self.max_life, self.max_birth, bandwidth=1).astype(np.float32)
        if self.persistence == 'pre':
            if self.transform is not None:
                sample = self.transform(Image.fromarray((255*sample).astype(np.uint8).transpose(1,2,0)))
        return sample, hs

    def __len__(self) -> int:
        return self.n_samples