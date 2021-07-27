# MetaReg
## Paper
Code release for "Meta-reweighted Regularization for Unsupervised Domain Adaptation"

This work proposes a novel regularization mechanism in deep UDA via meta-learning to guide the classifier to adapt better to the target domain.


## VisDA-2017 Experiment 

VisDA-2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

Run the following command in shell:

```shell
visda-2017 DANN+MetaReg

CUDA_VISIBLE_DEVICES=0 python MetaReg_dann.py /data1/TL/data/ -d VisDA2017 -s T -t V -a resnet50 -j 8 --epochs 30 --i 1000 --log_filename metareg-dann-visda --seed 2020

visda-2017 CDAN+MetaReg

CUDA_VISIBLE_DEVICES=0 python MetaReg_cdan.py /data1/TL/data/ -d VisDA2017 -s T -t V -a resnet50 -j 8 --epochs 30 --i 1000 --log_filename metareg-dann-visda --seed 2020

```

## Acknowledgement
Some codes in this project are borrowed from [Transfer-Learn](https://github.com/thuml/Transfer-Learning-Library) and [L2RW](https://github.com/danieltan07/learning-to-reweight-examples). We thank them for their excellent projects.

## Contact

If you have any problem about our code, feel free to contact

- wenxuanma@bit.edu.cn
- jm-zhang@bit.edu.cn

or describe your problem in Issues.