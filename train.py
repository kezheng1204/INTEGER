# -*- coding: future_fstrings -*-
import open3d as o3d  # prevent loading error

import sys
import json
import logging
import torch
from easydict import EasyDict as edict
import argparse

import warnings
import os
import os.path as osp

# Suppress annoying warnings from PyArrow and TensorFlow
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from lib.data_loaders import make_data_loader

from lib.trainer import get_trainer

from lib.utils import load_config
import datetime


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d %H:%M:%S", handlers=[ch]
)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def main(config, resume=False):
    seed = int(config.manual_seed)
    print(f"Use manual seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loader = make_data_loader(
        config,
        config.train_phase,
        config.batch_size,
        num_threads=config.train_num_thread,
    )

    if config.test_valid:
        val_loader = make_data_loader(
            config,
            config.val_phase,
            config.val_batch_size,
            num_threads=config.val_num_thread,
        )
    else:
        val_loader = None

    Trainer = get_trainer(config.trainer)
    trainer = Trainer(
        config=config,
        data_loader=train_loader,
        val_data_loader=val_loader,
    )

    trainer.train()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    print("Loading config from", config_path)
    
    config = load_config(config_path)

    config = edict(config)

    config.config_file = config_path


    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    config.out_dir = osp.join(config.out_dir, config.exp_name, f'{time}')

    return config



if __name__ == "__main__":
    logger = logging.getLogger()
    config = get_config()

    dconfig = vars(config)
    if config.resume_dir and not config.finetune_restart:
        resume_config = json.load(open(config.resume_dir + "/config.json", "r"))
        # for k in dconfig:
        #     if k not in ["resume_dir"] and k in resume_config:
        #         dconfig[k] = resume_config[k]
        dconfig["resume"] = resume_config["out_dir"] + "/checkpoint.pth"

    logging.info("===> Configurations")
    for k in dconfig:
        logging.info("    {}: {}".format(k, dconfig[k]))

    # Convert to dict
    config = edict(dconfig)
    main(config)
