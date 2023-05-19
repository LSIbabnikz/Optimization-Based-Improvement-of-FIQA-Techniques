
import os
import sys
import pickle
import random
import argparse

import torch
import wandb
import numpy as np
from torchvision.transforms import transforms as T

from train_regressor import train_network
from base_model.model import load_arcface


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":

    seed(42069)

    parser = argparse.ArgumentParser()
    parser.add_argument("-u_wandb", "--use_wandb", type=int, default=0, choices=[0, 1])
    parser.add_argument("-pn", "--project_name", type=str, default="label_training")
    parser.add_argument("-sl", "--save_location", type=str, required=True)
    parser.add_argument("-ql", "--quality_location", type=str, required=True)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-le", "--logs_per_epoch", type=int, default=20)
    parser.add_argument("-es", "--embedding_size", type=int, default=512)
    parser.add_argument("-vp", "--validation_percentage", type=float, default=.05)
    parser.add_argument("-e", "--epochs", default=1)
    parser.add_argument("-v", "--verbose", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    assert os.path.exists(args.quality_location), f"Quality scores location {args.save_location} does not exist!"
    assert args.batch_size > 0, f"Batch size should be a positive number"
    assert args.learning_rate > 0, f"Learning rate should be a positive number"
    assert 100 > args.logs_per_epoch > 0, f"Logs should be between (0, 100)"
    assert args.epochs > 0, f"Epochs should be a positive number"
    assert args.embedding_size > 0, f"Embedding size should be a positive number"

    model = load_arcface()
    trans = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    wandb_logger = None
    if args.use_wandb:
        wandb_logger = wandb.init(project=args.project_name)

    train_network(model, 
                  trans,
                  validation_percentage=args.validation_percentage,
                  quality_location=args.quality_location,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  epochs=args.epochs,
                  embedding_size=args.embedding_size,
                  use_wandb=bool(args.use_wandb), 
                  wandb_logger=wandb_logger, 
                  logs_per_epoch=args.logs_per_epoch,
                  save_location=args.save_location)
