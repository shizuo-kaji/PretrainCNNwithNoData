# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import utils
from model_select import model_select
from arguments import arguments
from dataset import DatasetFolderPH
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support,matthews_corrcoef

args = arguments()

if __name__== "__main__":
    
    print(args.val, args.img_size, args.crop_size)
    test_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.img_size > args.crop_size:
        test_transform.transforms.append(transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR))
    test_transform.transforms.append(transforms.CenterCrop(args.crop_size))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)
    test_dataset = datasets.ImageFolder(args.val, transform=test_transform)
    print(len(test_dataset), f"validation images loaded from {args.val}.")
    print(test_dataset.class_to_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = model_select(args, len(test_dataset.classes), renew_fc=False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        i=0
        for (data, targets) in test_loader:
            batch_size = targets.size(0)
            data = data.to(device,non_blocking=True)
            outputs = model(data)
            topk = min(outputs.shape[1],5)
            _, pred = outputs.topk(topk, 1, True, True)
            pred = pred.t().to('cpu') # top-k predictions
            correct = (pred == targets.unsqueeze(dim=0)).expand_as(pred)
            pred = pred.detach().numpy()
            if i==0:
                res = np.zeros(topk)
                prediction=[[] for k in range(topk)]
            for k in range(topk):
                res[k] += correct[:k].reshape(-1).float().sum(0, keepdim=True)
                prediction[k].extend(pred[k].tolist())
            i += batch_size

    fns = [f for f,t in test_dataset.imgs]
    truth = [t for f,t in test_dataset.imgs]
    #print(len(fns),len(truth),len(prediction[0]))
    print(confusion_matrix(truth,prediction[0]))
    print(classification_report(truth,prediction[0],digits=3))
    print(matthews_corrcoef(truth,prediction[0]))

    pd.DataFrame({'filename': fns, 'truth': truth, 'prediction_1': prediction[0], 'prediction_2': prediction[1]}).to_csv("predictions.csv")
