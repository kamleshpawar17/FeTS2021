import numpy as np
import glob
import os
import nibabel as nb

def np_dice_score_et(y_true, y_pred):
    true_l = np.array(y_true == 3).astype('float32')
    pred_l = np.array(y_pred == 3).astype('float32')
    den = (np.sum(true_l) + np.sum(pred_l))
    if den == 0:
        ds = 1.0
    else:
        ds = 2 * np.sum(true_l * pred_l) / den
    return ds

def np_dice_score_tc(y_true, y_pred):
    true_l1 = np.array(y_true == 1).astype('float32')
    true_l3 = np.array(y_true == 3).astype('float32')
    pred_l1 = np.array(y_pred == 1).astype('float32')
    pred_l3 = np.array(y_pred == 3).astype('float32')
    pred_tc = 1 - ((1 - pred_l1) * (1 - pred_l3))
    true_tc = 1 - ((1 - true_l1) * (1 - true_l3))
    den = (np.sum(true_tc) + np.sum(pred_tc))
    if den == 0:
        ds = 1.0
    else:
        ds = 2 * np.sum(true_tc * pred_tc) / den
    return ds

def np_dice_score_wt(y_true, y_pred):
    true_l1 = np.array(y_true == 1).astype('float32')
    true_l2 = np.array(y_true == 2).astype('float32')
    true_l3 = np.array(y_true == 3).astype('float32')
    pred_l1 = np.array(y_pred == 1).astype('float32')
    pred_l2 = np.array(y_pred == 2).astype('float32')
    pred_l3 = np.array(y_pred == 3).astype('float32')
    pred_wt = 1 - ((1 - pred_l1) * (1 - pred_l2))
    pred_wt = 1 - ((1 - pred_wt) * (1 - pred_l3))
    true_wt = 1 - ((1 - true_l1) * (1 - true_l2))
    true_wt = 1 - ((1 - true_wt) * (1 - true_l3))
    den = (np.sum(true_wt) + np.sum(pred_wt))
    if den == 0:
        ds = 1.0
    else:
        ds = 2 * np.sum(true_wt * pred_wt) / den
    return ds
    
def get_3d_image(dir_name, cont):
    '''
    takes the directory name, orientation, slice number and reads a slice, zero pad/crop and normalize 
    '''
    # ---- get slice for given contrast image ---- #
    fname = glob.glob(os.path.join(dir_name, cont))
    f = nb.load(fname[0])
    img = np.squeeze(f.get_fdata()).astype('float32')
    img = np.expand_dims(img, 0)
    return img
    
def get_3d_images_for_validation(dir_name):
    # ---- get images ---- #
    x_t1 = get_3d_image(dir_name, '*t1.nii.gz')
    x_t2 = get_3d_image(dir_name, '*t2.nii.gz')
    x_t1ce = get_3d_image(dir_name, '*t1ce.nii.gz')
    x_flair = get_3d_image(dir_name, '*flair.nii.gz')
    x_seg = get_3d_image(dir_name, '*seg.nii.gz').astype('int')
    x_seg[x_seg==4] = 3
    x_inp = np.concatenate((x_flair, x_t1, x_t2, x_t1ce), 0)
    return x_inp, np.squeeze(x_seg)

def get_3d_images_for_validation_nogt(dir_name):
    # ---- get images ---- #
    x_t1 = get_3d_image(dir_name, '*t1.nii.gz')
    x_t2 = get_3d_image(dir_name, '*t2.nii.gz')
    x_t1ce = get_3d_image(dir_name, '*t1ce.nii.gz')
    x_flair = get_3d_image(dir_name, '*flair.nii.gz')
    x_inp = np.concatenate((x_flair, x_t1, x_t2, x_t1ce), 0)
    return x_inp
    
def normalize_mean_std(x, axis=(-1, -2)):
    xmean = np.mean(x, axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    y = (x - xmean)/(1e-7+xstd)
    y[y>6.0] = 6.0
    y[y<-6.0] = -6.0
    return y

class image_zpad():
    '''
    This class zero pads the 3d input image to N=[w, h, sl]
    it takes 4d input 
    '''

    def __init__(self):
        self.r = None
        self.c = None
        self.s = None

    def zpad_to_N(self, img, N):
        self.n, self.r, self.c, self.s = img.shape
        self.zpad_n = N[0] - self.n
        self.zpad_r = N[1] - self.r
        self.zpad_c = N[2] - self.c
        self.zpad_s = N[3] - self.s
        self.zpad_n0 = int(np.ceil(self.zpad_n / 2.))
        self.zpad_n1 = int(np.floor(self.zpad_n / 2.))
        self.zpad_r0 = int(np.ceil(self.zpad_r / 2.))
        self.zpad_r1 = int(np.floor(self.zpad_r / 2.))
        self.zpad_c0 = int(np.ceil(self.zpad_c / 2.))
        self.zpad_c1 = int(np.floor(self.zpad_c / 2.))
        self.zpad_s0 = int(np.ceil(self.zpad_s / 2.))
        self.zpad_s1 = int(np.floor(self.zpad_s / 2.))
        img = np.pad(img, ((self.zpad_n0, self.zpad_n1), (self.zpad_r0, self.zpad_r1), (self.zpad_c0, self.zpad_c1), (self.zpad_s0, self.zpad_s1)),
                     'constant')
        return img

    def remove_zpad(self, img):
        img = img[self.zpad_n0:self.zpad_n0 + self.n, self.zpad_r0:self.zpad_r0 + self.r, self.zpad_c0:self.zpad_c0 + self.c,
              self.zpad_s0:self.zpad_s0 + self.s]
        return img

