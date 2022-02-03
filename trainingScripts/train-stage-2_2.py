import sys
sys.path.append('../')
import pytorch_lightning as pl
from data_module_np import PatchDataModule
from models import EncoderDecoderModuleFPN
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse

# ---- Arguments ---- #
ap = argparse.ArgumentParser()
ap.add_argument("-tp", "--train_path", type=str, default='../data/np/train/*.npy')
ap.add_argument("-vp", "--val_path", type=str, default='../data/np/val/*.npy')
ap.add_argument("-cp", "--checkpoint_path", type=str, default='../weights/stage2/FPN-se_resnet50/')
ap.add_argument("-lrf", "--lr_finder", type=int, default=0)
ap.add_argument("-bs", "--batch_size", type=int, default=16)
args = vars(ap.parse_args())

if __name__ == '__main__': 
    # ----- model defination ---- #
    model = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='se_resnet50',  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=0.0, lossWeight={'dice':1.0, 'CE':0.0}, lr=0.0001, red_lr_pla_factor=0.75, red_lr_pla_patience=15)

    # ---- data module ---- #
    train_path = args['train_path']
    val_path = args['val_path']
    dataModule = PatchDataModule(train_path, val_path, batch_size=args['batch_size'], num_workers=8, img_sz=224, data_aug=True, p=0.8, use_cache=False)
    
    # ---- check points, logs and trainer configuration --- #
    lr_finder_flag = args['lr_finder']
    default_root_dir = args['checkpoint_path']
    filepath = default_root_dir+'checkpoints/'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=filepath, save_top_k=1, every_n_val_epochs=1, period=1, verbose=True, monitor="val_loss_dice", mode="min")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator='ddp', callbacks=[lr_monitor, checkpoint_callback], default_root_dir=default_root_dir, gpus=torch.cuda.device_count(), max_epochs=40, benchmark=True, auto_lr_find=lr_finder_flag, resume_from_checkpoint=None, accumulate_grad_batches=1)
    
    # ---- learning rate finder ---- #
    if lr_finder_flag:
        # ----- Run learning rate finder ---- #
        print('learning rate before: ', model.lr)
        lr_finder = trainer.tuner.lr_find(model, dataModule, num_training=300)
        new_lr = lr_finder.suggestion()
        print('best lr: ', new_lr)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        plt.show()
    else:
        # --- Train ---- #
        trainer.fit(model, datamodule=dataModule)
   