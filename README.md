# Pretraining of CNN without data
by Shizuo Kaji, 2021

With mathematically generated images annotated with mathematically defined labels,
this code trains any CNN with no images nor labels, that is, completely in an unsupervised manner.
The obtained model can be fine-tuned for any tasks (transfer learning).

The code also works with any image dataset without labels; in this case, only labels are mathematically generated.


The pretraining is useful when ImageNet pretraining is not appropriate by some reasons such as fairness.
See
- Ninareh Mehrabi et al., A Survey on Bias and Fairness in Machine Learning, [arXiv:1908.09635](https://arxiv.org/abs/1908.09635)
- Maithra Raghu et al., Transfusion: Understanding Transfer Learning for Medical Imaging, NeurIPS 2019, [arXiv:1902.07208](https://arxiv.org/abs/1902.07208)
- Veronika Cheplygina, Cats or CAT scans: transfer learning from natural or medical image source datasets?, Current Opinion in Biomedical Engineering 9, [arXiv:1810.05444](https://arxiv.org/abs/1810.05444)
- Hirokatsu Kataoka et al., Pre-training without Natural Images, ACCV 2020, [arXiv:2101.08515](https://arxiv.org/abs/2101.08515)



Topological information encoded by the persistent homology is used as a pretraining task, 
so the trained CNN will be expected to focus more on the shape rather than the texture, contrasting to ImageNet pretrained models.
This may be helpful 
- for finetuning for tasks dealing with medical images, which are completely different from natural images contained in ImageNet.
- to gain robustness against adversarial attacks.

The implemented image generation is a simple FFT based one, but any image (synthesised or natural) can be used.

To sum up,
- No need for data collection
- No need for manual labelling
- Acquires robust image features based on topology




## Licence
MIT Licence

## Requirements
- a modern GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- PyTorch >= 1.8
- CubicalRipser: install by the following command

    % pip install git+https://github.com/shizuo-kaji/CubicalRipser_3dim

## Training data generation
Training data can be generated on the fly, but for the efficiency,
we recommend to precompute training images and their persistent homology.

    % python random_image.py -pb 0.5 -pc 0.5 -o random -n 50000

generates 50000 images (-n 50000) under the directory `random` (-o random). 
Half of them (-pc 0.5) are colour and the rest is grayscale.
Half of them (-pb 0.5) are binarised.

## Precomuting persistent homology
The following computes the persistent homology

    % python PHdict.py random -o PH_random -it jpg

of (the distance transform of the binarisation of) the jpeg images (-it jpg) under the directory `random` and outputs the results under `PH_random`.
Optionally, the gradient is taken before applying the distance transform when (-g) is specified.

Note that instead of synthesised images, we can use any image dataset (e.g., ImageNet).

## Model pre-training

    % python pretraining.py -o 'weights' --numof_classes=200 --label_type PH_hist -t 'random' -pd PH_random -u 'resnet50' --max_life 80 --max_birth 80 --bandwidth 2

You will find a pretrained weight file (e.g., `resnet50_epoch90.pth`) under the directory 'result/XX', where XX is automatically generated from the date.
Different types of persistent-homology-based labelling can be specified, for example, by (--label_type 'persistence_image').

If you wish to generate training images and labels on the fly (not efficient), do not specify the training images (i.e., do not use -t):

    % python pretraining.py -o 'weights' --numof_classes=200 --label_type PH_hist -pd PH_random -u 'resnet50' --alpha_range 0.01 1 --beta_range 0.5 2 -pc 0.5 -pb 0.5 -n 50000

The arguments (--alpha_range 0.01 1 --beta_range 0.5 2 -pc 0.5 -pb 0.5) are parameters for image generation. 
In each epoch, 50000 (-n 50000) images are generated.

## Model fine-tuning
The pretrained model can be fine-tuned for any downstream tasks.
The pretraining code saves the weights in a standard PyTorch model format, so you can use your own code to load the pretrained model.

Alternatively, we provide a code for finetuning

    % python finetuning.py -t 'data/CIFAR100/train' -val 'data/CIFAR100/test' -pw 'weights/XX/resnet50_epoch90.pth' -o 'result' -nc 100 -e 90

The CIFAR100 dataset can be obtained by the [script](https://github.com/chatflip/ImageRecognitionDataset) (included in this repository as well)

    % python util/ImageDatasetsDownloader.py --dataset CIFAR100

