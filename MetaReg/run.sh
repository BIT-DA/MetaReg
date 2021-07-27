# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python MetaReg_dann.py /data1/TL/data/ -d VisDA2017 -s T -t V -a resnet50 -j 8 --epochs 30 --i 1000 --log_filename metareg-dann-visda --seed 2020
CUDA_VISIBLE_DEVICES=0 python MetaReg_cdan.py /data1/TL/data/ -d VisDA2017 -s T -t V -a resnet50 -j 8 --epochs 30 --i 1000 --log_filename metareg-dann-visda --seed 2020
