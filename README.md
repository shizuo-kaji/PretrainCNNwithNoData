# Teaching topology to CNN by pretraining with persistent homology
by Shizuo Kaji, 2021

A scheme is presented for pretraining deep neural networks 
with synthetic images and mathematically defined labels that captures 
topological information in images.
The pretrained models can be finetuned for image classification tasks
to achieve an improved performance compared to those models trained from the scratch.

Convolutional neural networks, built upon iterative local operation, 
are better at learning local features of the image such as texture,
while our method provides an easy way to encourage them to learn global features.
Furthermore, our method requires no real images nor manual labels,
eliminating the concern related to data collection and annotation 
including the cost of manual labour and fairness issues.

With mathematically generated images annotated with mathematically defined labels,
this code trains any CNN with no images nor labels, that is, completely in an unsupervised manner.

This is a companion code for the paper "Teaching Topology to Neural Networks with Persistent Homology" by Shizuo Kaji and
Yohsuke Watanabe, in preparation.

Topological information encoded by the persistent homology is used as the regression target for the pretraining task, 
so the trained CNN will be expected to focus more on the shape rather than the texture, contrasting to ImageNet pretrained models.
This may be helpful 
- for finetuning for tasks dealing with medical images, which are completely different from natural images contained in ImageNet.
- to gain robustness against adversarial attacks.

The implemented image generation is a simple FFT based one, but any image (synthesised or natural) can be used.

To sum up,
- Acquires robust image features based on topology
- Works with virtually any neural network architectures
- No need for data collection nor manual labelling

Our pretraining scheme could be also useful when ImageNet pretraining is not appropriate by some reasons such as fairness.
See
- Ninareh Mehrabi et al., A Survey on Bias and Fairness in Machine Learning, [arXiv:1908.09635](https://arxiv.org/abs/1908.09635)
- Maithra Raghu et al., Transfusion: Understanding Transfer Learning for Medical Imaging, NeurIPS 2019, [arXiv:1902.07208](https://arxiv.org/abs/1902.07208)
- Veronika Cheplygina, Cats or CAT scans: transfer learning from natural or medical image source datasets?, Current Opinion in Biomedical Engineering 9, [arXiv:1810.05444](https://arxiv.org/abs/1810.05444)
- Hirokatsu Kataoka et al., Pre-training without Natural Images, ACCV 2020, [arXiv:2101.08515](https://arxiv.org/abs/2101.08515)


## Licence
MIT Licence

## Requirements
- a modern GPU
- Python 3: [Anaconda](https://anaconda.org) is recommended
- PyTorch >= 1.8
- tensorboard
- persim: install by the following command

    % pip install persim

- CubicalRipser: install by the following command

    % pip install git+https://github.com/shizuo-kaji/CubicalRipser_3dim

## Synthetic Training data generation
Training data can be generated on the fly, but for the efficiency,
we recommend to precompute training images and their persistent homology.

    % python random_image.py -pb 0.5 -pc 0.5 --alpha_range 0.01 1 --beta_range 0.5 2 -o random -n 200000 -nv 5000

generates 200000 images (-n 200000) under the directory `random` (-o random). 
Half of them (-pc 0.5) are colour and the rest are grayscale.
Half of them (-pb 0.5) are binarised.
alpha_range and beta_range are the frequency parameters.

## Precomputing persistent homology
The following computes the persistent homology

    % python PHdict.py random -o PH_random -it jpg

of (the distance transform of the binarisation of) the jpeg images (-it jpg) under the directory `random` and outputs the results under `PH_random`.
Optionally, the gradient is taken before applying the distance transform when (-g) is specified.

Note that we can use any image dataset (e.g., ImageNet) not restricted to synthetic images.

## Model pre-training

    % python training.py --numof_dims_pt=200 --label_type persistence_image -t 'random' -pd PH_random -u 'resnet50' --max_life 80 60 -lm 'pretraining'

You will find a pretrained weight file (e.g., `resnet50_pt_epoch90.pth`) under the directory 'result/XX', where XX is automatically generated from the date.
Different types of persistent-homology-based labelling (vectorisation) can be specified, for example, by (--label_type 'life_curve').
The 0-dimensional (resp. 1-dimensional) homology cycles with life time up to 80 (resp. 60) will be used for the labelling ('--max_life 80 60').
The label will be 200 dimensional (--numof_dims_pt=200).

If you wish to generate training images and labels on the fly (not efficient),

    % python training.py --numof_dims_pt=200 --label_type persistence_image -t 'generate' -u 'resnet50' --alpha_range 0.01 1 --beta_range 0.5 2 -pc 0.5 -pb 0.5 -n 50000 --max_life 80 80 -lm 'pretraining'

The arguments (--alpha_range 0.01 1 --beta_range 0.5 2 -pc 0.5 -pb 0.5) are parameters for image generation. 
In each epoch, 50000 (-n 50000) images are generated.

## Model fine-tuning
The pretrained model can be fine-tuned for any downstream tasks.
The pretraining code saves the weights in a standard PyTorch model format (e.g., 'result/XX/resnet50_pt_epoch90.pth'), 
so you can use your own code to load the pretrained model.

Our code can be used for finetuning as well.
(Note that our code does not aim at achieving high performance for downstream tasks.
It has very basic features which is suitable only for performance comparison):

    % python training.py -t 'data/CIFAR100/train' -val 'data/CIFAR100/test' -pw 'result/XX/resnet50_pt_epoch90.pth' -o 'result' -e 90 -lm 'finetuning'

The CIFAR100 dataset can be obtained by the [script](https://github.com/chatflip/ImageRecognitionDataset) (included in this repository as well)

    % python util/ImageDatasetsDownloader.py --dataset CIFAR100


## Experiments on the accuracy improvement in image classification tasks

![C100](https://github.com/shizuo-kaji/PretrainCNNwithNoData/blob/master/demo/C100.jpg?raw=true)

The graph shows the classification accuracies of the CIFAR100 dataset 
with models pretrained in different datasets and tasks.
Note that the purpose of the experiment is to show the effectiveness of our method but not to maximise the performance.
So the hyper-parameters are fixed (not optimised) and the performances with different pretraining conditions are compared.

The naming convention used in the graphs is PROBLEM_DATASET_TASK, where
- PROBLEM is either C100 (CIFAR100) or OMN (Omniglot).
- scratch indicates without any pretraining (random initialisation)
- DATASET indicates the dataset used for pretraining. 
IMN is the usual ImageNet-1k dataset, 
FB1000 (FB10000) is the FractalDB-1k (FractalDB-10k) dataset downloadable at [here](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/),
C100 is the CIFAR100 dataset,
and gen is the synthetic random image dataset generated as above.
- TASK indicates the task used for pretraining.
label means the classification of the labels that come with the dataset.
PH means the regression of persistent homology as above.

For the natural image classification (CIFAR100), the ImageNet pretrained model outperforms others by a large margin.
Still, it is notable that our method (gen_PH) is far better than learning from scratch.
