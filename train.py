import os
import argparse
from tkinter import N
import numpy as np
import time
import torch
import torch.utils.data as data
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from data import get_loader
from protonet import HierarchicalProtoNet


def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 

def train(
    model_class, 
    train_loader, 
    val_loader,
    ckpt_dir, 
    **kwargs):
    model = model_class(**kwargs)
    
    trainer = pl.Trainer(
        default_root_dir=os.path.join(ckpt_dir, model_class.__name__),
        gpus=1,
        max_epochs=100,
        logger=get_wandb_logger(model),

        callbacks=[
            ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="Best-{lr:.4f}-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            save_last= True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min", 
            verbose=True,
            ),
            LearningRateMonitor("step"),
            EarlyStopping(
                patience=5,
                min_delta=0.00,
                monitor="val_loss",
                mode="min",
                verbose=True,
            )
        ],
        num_sanity_val_steps=0
    )
    trainer.logger._default_hp_metric = None
    trainer.fit(model, train_loader, val_loader)

    model = model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return model


if __name__ == '__main__':
    pl.seed_everything(42)
    date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    parser = argparse.ArgumentParser(description='Parser for non-/hierarchical prototypical networks.')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--way', type=int, required=True, default=5)
    parser.add_argument('--shot', type=int, required=True, default=15)
    parser.add_argument('--query', type=int, required=True, default=30)
    parser.add_argument('--taxonomy', type=str, required=True, default='AS', help='Provide taxonomy among AS, VR, SG.')
    parser.add_argument('--alpha', type=float, default=-0.5)
    parser.add_argument('--lr', type=float, default=1e-03)

    args = parser.parse_args()
    
    n_class = args.way
    n_shot = args.shot
    n_query = args.query
    height = args.height
    taxonomy = args.taxonomy
    loss_alpha = args.alpha
    lr = args.lr


    train_loader, _ = get_loader(phase='train',
                                 n_class=n_class, n_shot=n_shot, n_query=n_query)

    val_loader, _ = get_loader(phase='val',
                               n_class=n_class, n_shot=n_shot, n_query=n_query)

    wandb.finish()
    wandb.init(project=f'exp-{taxonomy}',
            #    entity= wandb user ID,
               name=f'height{height}_n{n_class}k{n_shot}q{n_query}_lr{lr}_a{loss_alpha} - {date}'
               )

    print("============================================================================")
    print("TRAIN", f"{height}-height, {n_class}-class, {n_shot}-shot, {n_query}-query")

    train(model_class=HierarchicalProtoNet,
          train_loader=train_loader, 
          val_loader=val_loader,
          n_class=n_class,
          n_shot=n_shot,
          n_query=n_query,
          height=height,  
          loss_alpha=loss_alpha,
          taxonomy=taxonomy,
          lr=lr,
          ckpt_dir = f'./exp/{taxonomy}/H{height}_n{n_class}k{n_shot}q{n_query}_a{loss_alpha}_lr{lr}_{date}'
        )

      
