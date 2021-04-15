# Generalized-ODIN-TF
TensorFlow 2 implementation of the paper [Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data](https://arxiv.org/abs/2002.11297).

Detecting out-of-distribution (OOD) data is a challenging problem for deep neural networks to tackle especially when they weren't exposed to OOD data. One way to solve this is to [expose networks](https://arxiv.org/abs/1812.04606) to OOD data _during_ its training. But this can become a brittle approach when the space of the OOD data gets larger for a network. What if we present a network with OOD data it _hasn't_ been exposed to during its training? 

Therefore, we need a better way to deal with the problem. Generalized ODIN (**O**ut-of-**DI**stribution detector for **N**eural networks) is a good first step toward that. 

## Organization of the files

```shell
├── Baseline.ipynb: Trains a ResNet20 model on the CIFAR-10 dataset (in-distribution dataset). We will consider this to be our baseline model.
├── Calculate_Epsilon.ipynb: Searches for the best epsilon (perturbation magnitude) as proposed in the paper. 
├── Evaluation_OOD.ipynb: Evaluates the baseline model as well as the Generalized ODIN model.
├── Generalized_ODIN.ipynb: Trains the Generalized ODIN model on the CIFAR-10 dataset. 
└── scripts
    ├── metrics.py: Utilities for evalutation metrics (AUROC and TNR@TPR95)
    ├── resnet20.py: ResNet20 model utilities. 
    └── resnet20_odin.py: ResNet20 with Generalized ODIN utilities. 
```

_TNR: True Negative Rate, TPR: True Positive Rate_

## Task of interest

Train a model on the CIFAR-10 dataset (in-distribution dataset) in a way that maximizes its capability to detect OOD samples. This project uses the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/) for the OOD samples.

## Main differences in the implementation

* The authors use ResNet34. In this project, ResNet20 has been used. 
* The learning rate schedule goes like following: Decay by a factor of 0.1 at 25%, 50%, and 75% of the total training epochs. 
* DeConf-I's been used to calculate `h(x)`. Refer to the paper for more details (Section 3.1.1).

## Results

<p align="center">

|                  	| Train Top-1 	| Test Top-1 	| AUROC 	| TNR@TPR95 	|
|------------------	|:-----------:	|:----------:	|:-----:	|:---------:	|
| Generalized ODIN 	|    99.46    	|    91.42   	| **92.15** 	|   **54.18**   	|
|     Baseline     	|    99.58    	|    90.7    	| 91.14 	|   40.53   	|

</p>

## TODO

- [ ] Add WideResNet-28-10 results

## How to use these models to detect OOD samples?

Take the output of the ODIN and see if it crosses a threshold. If it does then the corresponding samples IID otherwise OOD.

Here's an advice I got from Yen-Chang Hsu (first author of the paper):

> The selection of thresholds is application-dependent. It will still rely on having a validation set (which includes in-distribution and optionally OOD data) for an application. One way is to select the threshold is pick one that makes TPR=95%.

## Pre-trained models

Available [here](https://github.com/sayakpaul/Generalized-ODIN-TF/releases/download/v1.0.0/models.tar.gz).

## Acknowledgements

* Thanks to Yen-Chang Hsu for providing constant guidance. 
* Thanks to the [ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP support. 

## Paper citation

```
@INPROCEEDINGS{9156473,
  author={Y. -C. {Hsu} and Y. {Shen} and H. {Jin} and Z. {Kira}},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Generalized ODIN: Detecting Out-of-Distribution Image Without Learning From Out-of-Distribution Data}, 
  year={2020},
  volume={},
  number={},
  pages={10948-10957},
  doi={10.1109/CVPR42600.2020.01096}}
```
