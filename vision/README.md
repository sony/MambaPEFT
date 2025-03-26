# MambaPEFT for Vision Tasks

This directory includes the MambaPEFT for reproducing the results in our paper. 

## Setup

Install dependencies. Our code is tested on CUDA 11.7, Pytorch 2.0.0, and Python 3.8.  
For example,
```sh
pip install torch==2.0.0 torchvision==0.15.1
```
Then,
```sh
pip install -U openmim && mim install "mmpretrain==1.2"
pip install optuna
# install Mamba locally (small modified version from the original Mamba for Vim)
pip install personal_lib/external_packages/mamba-1p1p1 --user 
```

### If you want to use docker instead of venv/conda...
```sh
cd docker/cuda11/ 
nvidia-docker build -t mambapeft:cuda117 .
nvidia-docker run -itd --shm-size=16gb -h docker --name trainer1 -v [PATH-to-vision-DIRECTORY]:/work -v [PATH-to-YOUR-DATA-DIRECTORY]:/data mambapeft:cuda117
docker exec -it trainer1 /bin/bash

# additional install in docker env. (small modified version from the original Mamba for Vim)
pip install personal_lib/external_packages/mamba-1p1p1 --user 
```

## Datasets and Weights of base-models 

Download the VTab-1K dataset.
```sh
# Change the "BASEDIR=/data" to your directory
bash tools/download_vtab.sh
```

Download the Vim's pre-trained weight.
```sh
# from
https://huggingface.co/hustvl/Vim-small-midclstok
# and place it as
work_dirs/weights/backbone/vim/vim_s_midclstok_80p5acc.pth
```

## Fine-tuning and Evaluation
#### LoRAp(X)
```sh
CONFIG_NAME=mmpretrain/vim/vtab1k/1_small/2_small_lorap_X_64
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 \
               patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist \
               dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh mmpretrain configs/${CONFIG_NAME}.py 1 \
            --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR} \
            --cfg-options sub_dataset_name=${DATASET}
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh mmpretrain configs/${CONFIG_NAME}.py \
            work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 1  \
            --cfg-options sub_dataset_name=${DATASET} | \
            tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result_${DATASET}.txt
    done

```

#### Affix-tuning
```sh
CONFIG_NAME=mmpretrain/vim/vtab1k/1_small/4_small_affixtune_prefix3dual_wo_project
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 \
               patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist \
               dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh mmpretrain configs/${CONFIG_NAME}.py 1 \
            --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR} \
            --cfg-options sub_dataset_name=${DATASET}
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh mmpretrain configs/${CONFIG_NAME}.py \
            work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 1  \
            --cfg-options sub_dataset_name=${DATASET} | \
            tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result_${DATASET}.txt
    done

```

#### Additional-scan
```sh
CONFIG_NAME=mmpretrain/vim/vtab1k/1_small/4_small_addiscan_prefix_6_copyfirst
CHECKPOINT_DIR=`date "+%Y%m%d_%H%M%S_%N"`
for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 \
               patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist \
               dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh mmpretrain configs/${CONFIG_NAME}.py 1 \
            --work-dir work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR} \
            --cfg-options sub_dataset_name=${DATASET}
        CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh mmpretrain configs/${CONFIG_NAME}.py \
            work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/last_checkpoint 1  \
            --cfg-options sub_dataset_name=${DATASET} | \
            tee work_dirs/${CONFIG_NAME}/${CHECKPOINT_DIR}/result_${DATASET}.txt
    done

```

## Hybrid PEFT Search
Our search method to find the better combination of PEFT methods and their hyperparameters is two steps.

### 1. Find the better combination with minimal trainable parameters in each PEFT methods.
Please fit device-ids to your environment. If you set as "0 0 1 1", then two parallel processes use the same GPUs. 
```sh
python tools/tuning_tools/train_time_optuna_tuning_vtab.py \
    configs/mmpretrain/vim/vtab1k/1_small/5_small_peft_search_cfg_step1.py \
    --device-ids 0 1 2 3 \
    --trials 100 \
    --original-command "python tools/train.py mmpretrain configs/mmpretrain/vim/vtab1k/1_small/5_small_peft_search_step1.py"
```

### 2. Optimize hyperparameters of the chozen PEFT methods (and potentially remove one PEFT).
Please fix '5_small_peft_search_cfg_step2.py' based on your best result with the step 1. Then, run as
```sh
python tools/tuning_tools/train_time_optuna_tuning_vtab.py \
    configs/mmpretrain/vim/vtab1k/1_small/5_small_peft_search_cfg_step2.py \
    --device-ids 0 1 2 3 \
    --trials 100 \
    --original-command "python tools/train.py mmpretrain configs/mmpretrain/vim/vtab1k/1_small/5_small_peft_search_step2.py"
```
You can repeat the second step multiple times but it did not improve accuracy in our experiments. 


## Acknowledgements

Our code is based on [Vim](https://github.com/hustvl/Vim), [MMPreTrain](https://github.com/open-mmlab/mmpretrain), [VMamba](https://github.com/MzeroMiko/VMamba), and [PETL-ViT](https://github.com/JieShibo/PETL-ViT).