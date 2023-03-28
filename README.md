# Hierarchical Classification for Environmental Sounds with Few-shot Learning

PyTorch implementation of "Hierarchical Classification for Environmental Sounds with Few-shot Learning"

![main](https://user-images.githubusercontent.com/116733923/201276769-dda3e4f8-24d6-4b17-89de-84f43cb17df5.png)

## Requirements
1. Install python and PyTorch:
```
  python==3.9.13
  pytorch-lightning==1.6.1
  torch==1.11.0
```
2. Other requirements:
```
pip install -r requirements.txt
```

## Training 
1. Reproduce the data, FSD-MIX-CLIPS, by following the instruction in original repo. [link](https://github.com/wangyu/rethink-audio-fsl#dataset)

2. Preprocessing audio
python preprocessing.py

3. Trainining options
Example of training 5-way 15-shot 30-query on hierarchical Prototypical Networks with level-2 hierarchy.
```
python train.py --height 2 --way 5 --shot 15 --query 30 --taxonomy 'AS' 
```

## Testing
Refer to test.py to estimate and compare performances. (WIP)
