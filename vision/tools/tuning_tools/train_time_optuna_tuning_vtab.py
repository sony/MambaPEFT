"""
Train 時にハイパラを探索する場合に使用
"""
import optuna
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from functools import partial
import subprocess as sp
import json
import torch.distributed as dist
import torch.multiprocessing as mp
from optuna.trial import TrialState
import torch
import logging
import sys
import time
import pandas as pd
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description='test time param tuning')
    parser.add_argument('config', help='tuning config file path')
    parser.add_argument(
        '--direction', default="maximize")
    parser.add_argument(
        '--trials', type=int, default=100)
    parser.add_argument(
        "--device-ids",
        "-d",
        nargs="+",
        type=int,
        default=[0],
        help="Specify device_ids",
    )
    parser.add_argument(
        "--sub-dataset-names",
        "-s",
        nargs="+",
        type=str,
        default=['clevr_count', 'dmlab', 'oxford_iiit_pet', 'patch_camelyon', 'sun397'],
        help="select from cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele",
    )
    parser.add_argument('--original-command', help='original test command.')
    parser.add_argument("--master-port", type=str, default="12355", help="Now it is not used")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    search_setting = cfg.search_setting

    ori_command = args.original_command
    
    config_file = ori_command.split("train.py ")[-1].split(" ")[1]
    config_file = config_file.replace("configs/", "work_dirs/").replace(".py", "")


    # conn = sqlite3.connect("test.db")
    # conn.close()
    device_ids = args.device_ids
    world_size = max(len(device_ids), 1)
    manager = mp.Manager()
    return_dict = manager.dict()
    print("a")
    mp.spawn(
        run_optimize,
        args=(world_size, device_ids, return_dict, args.master_port, args.direction, search_setting, ori_command, config_file, cfg.metric_prefix, args.trials, args.sub_dataset_names),
        nprocs=world_size,
        join=True,
    )
    study = return_dict["study"]

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))


    save_dict = {}
    print("Best trial:")
    trial = study.best_trial
    save_dict[cfg.metric_prefix] = trial.value
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    save_dict["best_params"] = {}
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        save_dict["best_params"][key] = value

    with open(osp.join(work_dir, "optim_resuls.json"), 'w') as f:
        json.dump(save_dict, f, indent=2)


def run_optimize(rank, world_size, device_ids, return_dict, master_port, direction, search_setting, ori_command, config_file, metric_prefix, n_trials, sub_dataset_names):
    device = device_ids[rank]
    print(f"Running basic DDP example on rank {rank} device {device}.")
    os.makedirs(config_file, exist_ok=True)
    storage_name = f"sqlite:////work/{config_file}/optuna.db"
    if rank == 0:
        print("a")
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study = optuna.create_study(direction=direction, storage=storage_name, study_name=config_file, load_if_exists=True)
        study.optimize(
            partial(objective, device_id=device, search_setting=search_setting, ori_command=ori_command, config_file=config_file, metric_prefix=metric_prefix, sub_dataset_names=sub_dataset_names),
            n_trials=n_trials//len(device_ids),
            n_jobs=1)
        return_dict["study"] = study
    else:
        time.sleep(5)
        study = optuna.load_study(storage=storage_name, study_name=config_file)
        study.optimize(
            partial(objective, device_id=device, search_setting=search_setting, ori_command=ori_command, config_file=config_file, metric_prefix=metric_prefix, sub_dataset_names=sub_dataset_names),
            n_trials=n_trials//len(device_ids),
            n_jobs=1)

def get_metric(filepath, prefix):
    with open(filepath, 'r') as f:
        data_lines = f.readlines()
    data_lines = data_lines[-70:] #ラスト70行だけから探索

    for data in data_lines:
        if f"{prefix} " in data:
            value = data.split(f"{prefix} ")[1].split(" ")[0]
            try:
                value = float(value)
            except:
                print(value)
                raise ValueError(filepath, data)
            break
    return value

def objective(trial, device_id, search_setting, ori_command, config_file, metric_prefix, sub_dataset_names):
    # trial = optuna.integration.TorchDistributedTrial(single_trial)

    ori_command = f"CUDA_VISIBLE_DEVICES={device_id} "+ori_command
    
    work_dir = os.path.join(config_file, pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f"))    
    ori_command += " --work-dir " + work_dir

    ori_command += " --cfg-options "

    os.makedirs(work_dir, exist_ok=True)
    
    key_param_dict = {}
    for key, setting in search_setting.items():
        
        if "origin" in setting.keys():
            val = key_param_dict[setting["origin"]]
        elif "categories" in setting.keys():
            if isinstance(setting.categories[0], list):
                val=""
                for i in range(len(setting.min)):
                    val_ = trial.suggest_categorical(key+f".{i}", setting.categories[i])
                    val += str(val_)+","
                val = val[:-1]
            else:
                val = trial.suggest_categorical(key, setting.categories)
            key_param_dict[key] = val
        elif "add_categories" in setting.keys():
            val = trial.suggest_categorical(key, setting.add_categories)
            if not (val == "_not_add"):
                key_param_dict[val] = True
                key = val
                val = True
        elif "remove_categories" in setting.keys():
            val = trial.suggest_categorical(key, setting.remove_categories)
            if not (val == "_not_remove"):
                key_param_dict[val] = False
                key = val
                val = False
        elif "int_min" in setting.keys():
            if isinstance(setting.int_min, list):
                val=""
                for i in range(len(setting.int_min)):
                    val_ = trial.suggest_int(key+f".{i}", setting.int_min[i], setting.int_max[i], step=setting.get("step", 1), log=setting.get("log", False))
                    val += str(val_)+","
                val = val[:-1]
            else:
                val = trial.suggest_int(key, setting.int_min, setting.int_max, step=setting.get("step", 1), log=setting.get("log", False))
            key_param_dict[key] = val
        else:
            if isinstance(setting.min, list):
                val=""
                for i in range(len(setting.min)):
                    val_ = trial.suggest_float(key+f".{i}", setting.min[i], setting.max[i], step=setting.get("step", None), log=setting.get("log", False))
                    val += str(val_)+","
                val = val[:-1]
            else:
                val = trial.suggest_float(key, setting.min, setting.max, step=setting.get("step", None), log=setting.get("log", False))
            key_param_dict[key] = val

        ori_command += f"{key}={val} "

    metric_all = 0
    for sub_dataset_name in sub_dataset_names:
        ori_command_per_db = ori_command + f"sub_dataset_name={sub_dataset_name} "
        sp.call(ori_command_per_db, shell=True) # Train

        # with open(save_file) as f:
        #     last_saved = f.read().strip()
        
        ori_command_per_db = ori_command_per_db.replace("train.py", "test.py")
        ori_command_per_db = ori_command_per_db.split("--work-dir")[0] + os.path.join(work_dir, "last_checkpoint") + " --work-dir" + ori_command_per_db.split("--work-dir")[1]
        ori_command_per_db += f" | tee {work_dir}/optimizing_step_trial{trial.number}_{sub_dataset_name}.txt"
        
        sp.call(ori_command_per_db, shell=True) # Test

        metric_val = get_metric(os.path.join(work_dir, f"optimizing_step_trial{trial.number}_{sub_dataset_name}.txt"), metric_prefix)
        metric_all += metric_val
    
    # if trial.number % 4 == 0:
    #     shutil.copyfile("/optuna.db", os.path.join(work_dir, "optuna.db"))
    
    return metric_all/(len(sub_dataset_names))

if __name__ == "__main__":

    main()