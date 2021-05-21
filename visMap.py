# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from model_select import model_select
from arguments import arguments

args = arguments()

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    utils.save_image(grid, 'conv1_filters.png' ,nrow =nrow)


if __name__ == "__main__":
    layer = 1
    model = model_select(args)
    kernels = model.conv1.weight.detach().clone()
    print(kernels.size())

    visTensor(kernels, ch=0, allkernels=False)

