# Multi-directional and Multi-constraint Learning Network for Aerial Imagery Semantic Segmentation

![Network](./images/network.png)


## Folder Structure

Prepare the following folders to organize this repo:
```none
├── MMLN (code)
├── pretrain_weights (save the pretrained weights like vit, swin, etc)
├── model_weights (save the model weights)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   ├── uavid
│   ├── vaihingen
│   ├── potsdam 
```
## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r GeoSeg/requirements.txt
```

## Pretrained Weights

[Quark Netdisk](https://pan.quark.cn/s/6ad115af5302) : rEsN

## Data Preprocessing

[Quark Netdisk](https://pan.quark.cn/s/dd067d024b07) : YxVA


## Training

```
python MMLN/train_supervision.py -c MMLN/config/uavid/***.py
```
Use different **config** to train different models.

## Validation

For example:
```
python MMLN/loveda_test.py -c MMLN/config/loveda/***.py -o fig_results/loveda/*** --rgb --val -t 'd4'
```

## Testing

**LoveDA**
```
python MMLN/loveda_test.py -c MMLN/config/loveda/***.py -o fig_results/loveda/*** -t 'd4'
```

**UAVid**
```
python MMLN/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c MMLN/config/uavid/***.py \
-o fig_results/uavid/*** \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```

## Inference on huge remote sensing image
```
python MMLN/inference_huge_image.py \
-i data/vaihingen/test_images \
-c GeoSeg/config/vaihingen/***.py \
-o fig_results/vaihingen/*** \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```


## Reproduction Results
|    Method     |  Dataset  |  F1   |  OA   |  mIoU |model_weight|
|:-------------:|:---------:|:-----:|:-----:|------:|---------:|
|  MAMLN   | Vaihingen | 91.18 | 91.63 | 84.02 |[Quark Netdisk](https://pan.quark.cn/s/32266ef9bf21) : 1f29|
|  MAMLN   |  Potsdam  | 93.37 | 91.95 | 87.77 |[Quark Netdisk](https://pan.quark.cn/s/dafc3fb8a887) : 3DTA|
|  MAMLN   |  LoveDA   |   -   |   -   | 53.11 |[Quark Netdisk](https://pan.quark.cn/s/ecf76810a0f1) : RC25|
|  MAMLN   |   UAVid   |   -   |   -   | 70.51 |[Quark Netdisk](https://pan.quark.cn/s/81bf81e07b54) : ejQa|


Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.

