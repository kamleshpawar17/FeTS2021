import pytorch_lightning as pl
from torch import nn
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional  as F
from segmentation_models_pytorch.losses import DiceLoss

class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.decoder_attention_type = decoder_attention_type
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.Unet(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans, decoder_attention_type=self.decoder_attention_type)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss
        
class EncoderDecoderModuleUnetPP(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.decoder_attention_type = decoder_attention_type
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.Unet(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans, decoder_attention_type=self.decoder_attention_type)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss

class EncoderDecoderModuleMAnet(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.decoder_attention_type = decoder_attention_type
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.MAnet(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        #loss_dice = self.dice_loss(xtarget, xinput)
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss
     
class EncoderDecoderModuleFPN(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.FPN(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        #loss_dice = self.dice_loss(xtarget, xinput)
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss

class EncoderDecoderModulePSPNet(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.PSPNet(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        #loss_dice = self.dice_loss(xtarget, xinput)
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss

class EncoderDecoderModulePAN(pl.LightningModule):
    def __init__(self, in_chans: int=4, out_chans: int=4, encoder_name="resnet34",  encoder_weights="imagenet", decoder_attention_type='scse', weight_decay=4e-4, lossWeight={'dice':1.0, 'CE':0.0}, lr=1e-4, red_lr_pla_factor=0.9, red_lr_pla_patience=3):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.weight_decay = weight_decay
        self.lossWeight = lossWeight
        self.lossCE = torch.nn.CrossEntropyLoss()
        self.lossDice = DiceLoss(mode='multiclass', log_loss=True)
        self.lr = lr
        self.red_lr_pla_factor = red_lr_pla_factor
        self.red_lr_pla_patience = red_lr_pla_patience
        self.model = smp.PSPNet(encoder_name=encoder_name,  encoder_weights=encoder_weights, in_channels=self.in_chans, classes=self.out_chans)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.red_lr_pla_factor, patience=self.red_lr_pla_patience, verbose=True), 'monitor': 'val_loss'}}

    def forward(self, x):
        x_hat = self.model(x)    
        return x_hat
    
    def loss(self, xinput, xtarget):
        #loss_dice = self.dice_loss(xtarget, xinput)
        loss_dice = self.lossDice(xinput, torch.squeeze(xtarget, 1))
        loss_ce = self.lossCE(xinput, torch.squeeze(xtarget, 1))
        loss = self.lossWeight['dice']*loss_dice + self.lossWeight['CE']*loss_ce
        return loss, loss_dice, loss_ce
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self(x)   
        # --- loss ---- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        self.log_dict({'train_loss': loss,  'train_loss_dice': loss_dice, 'train_loss_ce': loss_ce})
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # --- forward pass ---- #
        x_hat = self.forward(x)   
        # --- losses ----- #
        loss, loss_dice, loss_ce = self.loss(x_hat, y)
        # --- loss logs --- #
        self.log_dict({'val_loss': loss, 'val_loss_dice': loss_dice, 'val_loss_ce': loss_ce})
        return loss