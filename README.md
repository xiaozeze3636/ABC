# ABC
Self-supervised trajectory representation learning by integrating multi-scale wavelet transform

Python version 3.7 and PyTorch version 1.7.1

numpy==1.19.4
torch==1.7.1
scipy==1.5.3
pandas==1.1.5
tensorboard==2.4.1
scikit-learn==0.24.0
tqdm==4.60.0
matplotlib==3.5.1
protobuf==3.20.1


The dataset was sourced from Self-supervised Trajectory Representation Learning with Temporal Regularities and Travel Semantics's study (2023), accessible at 
https://github.com/aptx1231/START?tab=readme-ov-file.


python run_model.py --model BERTContrastiveLM --dataset porto --config porto --gpu_id 0 --mlm_ratio 0.6 --contra_ratio 0.4 --split true --distribution geometric --avg_mask_len 2 --out_data_argument1 trim --out_data_argument2 shift


python run_model.py --model LinearETA --dataset porto --gpu_id 0 --config porto --pretrain_path libcity/cache/337030/model_cache/337030_BERTContrastiveLM_porto.pt

python run_model.py --model LinearClassify --dataset porto --gpu_id 0 --config porto --pretrain_path libcity/cache/337030/model_cache/337030_BERTContrastiveLM_porto.pt
