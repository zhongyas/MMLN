
## Folder Structure

Prepare the following folders to organize this repo:
```none
├── MAMLN (code)
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
python MAMLN/train_supervision.py -c MAMLN/config/uavid/***.py
```
Use different **config** to train different models.

## Validation

For example:
```
python MAMLN/loveda_test.py -c MAMLN/config/loveda/***.py -o fig_results/loveda/*** --rgb --val -t 'd4'
```

## Testing

**LoveDA**
```
python MAMLN/loveda_test.py -c MAMLN/config/loveda/***.py -o fig_results/loveda/*** -t 'd4'
```

**UAVid**
```
python MAMLN/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c MAMLN/config/uavid/***.py \
-o fig_results/uavid/*** \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```

## Inference on huge remote sensing image
```
python MAMLN/inference_huge_image.py \
-i data/vaihingen/test_images \
-c GeoSeg/config/vaihingen/***.py \
-o fig_results/vaihingen/*** \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```



## Reproduction Results
