import sys
sys.path.append('../../common/downloaded/segmentationPytorch/')
from models import EncoderDecoderModule, EncoderDecoderModuleFPN, EncoderDecoderModuleUnetPP
import numpy as np
import glob
from predict_utils import get_3d_images_for_validation_nogt, normalize_mean_std, image_zpad
import torch
import os
import matplotlib.pyplot as plt
import time
from patch_utils import find_bbox
import cv2
from albumentations.augmentations.geometric.resize import Resize
import nibabel as nib
from skimage.morphology import binary_dilation
from skimage.measure import label, regionprops
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def remove_small_blobs(img_seg, perThrshld=10.0):
    # --- convert image to binary --- #
    img_bin = img_seg.copy()
    img_bin[img_bin>0] = 1 
    # --- find blobs ---- #
    label_image, n = label(img_bin, return_num=True, background=0)
    regions = list(regionprops(label_image))
    # --- find the area of the biggest blob --- #
    max_area = 0.0
    for k, region in enumerate(regions):
        if region.area>max_area:
            max_area = region.area
    # --- refine labels based on area of the blob --- #
    img_seg_refined = img_seg.copy() 
    for k in range(n):
        area_per = regions[k].area/float(max_area)*100
        if area_per < perThrshld:
            img_seg_refined[label_image==k+1] = 0
    return img_seg_refined

def predict_3d(model, x_inp):
    output = np.zeros(x_inp.shape) # (256, 4, 256, 160)
    for k in range(x_inp.shape[0]):
        x = torch.tensor(x_inp[k:k+1]).to(device)
        y = torch.nn.functional.softmax(model.forward(x), dim=1)
        output[k] = y.cpu().detach().numpy() 
    return output
    

def predict_3d_sagcorax_stage1(model, x_inp_orig):
    zpad = image_zpad()
    x_inp_orig = zpad.zpad_to_N(x_inp_orig, [4, 256, 256, 160])
    # --- sag --- #
    x_inp = np.transpose(x_inp_orig, (1, 0, 2, 3)) # (256, 4, 256, 160)
    x_inp = normalize_mean_std(x_inp, axis=(-1, -2))
    output = predict_3d(model, x_inp)
    y_prob = np.transpose(output, (1, 0, 2, 3))
    # --- cor --- #
    x_inp = np.transpose(x_inp_orig, (2, 0, 1, 3))
    x_inp = normalize_mean_std(x_inp, axis=(-1, -2))
    output = predict_3d(model, x_inp)
    y_prob += np.transpose(output, (1, 2, 0, 3))
    # --- axial --- #
    x_inp = np.transpose(x_inp_orig, (3, 0, 1, 2))
    x_inp = normalize_mean_std(x_inp, axis=(-1, -2))
    output = predict_3d(model, x_inp)
    y_prob += np.transpose(output, (1, 2, 3, 0))  
    # -- proeb to label ---- #
    y_prob = zpad.remove_zpad(y_prob)
    y_pred = np.squeeze(np.argmax(y_prob, 0))
    return y_pred, y_prob

def resize_multislc(x, img_h, img_w, interp=cv2.INTER_LINEAR):
    y = np.zeros((x.shape[0], img_h, img_w))
    resize = Resize(img_h, img_w, interp)
    for k in range(x.shape[0]):
        y[k] = resize(image=x[k])['image']
    return y
               
def predict_3d_stage2(model, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25):
    # x_inp: (240, 4, 240, 160)
    # y_pred_s1: (240, 240, 160)
    output = np.zeros(x_inp.shape) # (240, 4, 240, 160)
    for k in range(x_inp.shape[0]):
        if np.sum(y_pred_s1[k]) > 0:
            bbox = find_bbox(y_pred_s1[k], minbbox=minbbox, bboxper=bboxper)
            minr, minc, maxr, maxc = bbox[0]
            inp_img = resize_multislc(x_inp[k, :, minr:maxr, minc:maxc], img_sz, img_sz, interp=cv2.INTER_NEAREST)
            inp_img = normalize_mean_std(np.expand_dims(inp_img, 0), axis=(-1, -2))
            x = torch.tensor(inp_img.astype('float32')).to(device)
            y = torch.nn.functional.softmax(model.forward(x), dim=1)
            out_patch = y.cpu().detach().numpy() 
            output[k, :, minr:maxr, minc:maxc] = resize_multislc(np.squeeze(out_patch), maxr-minr, maxc-minc, interp=cv2.INTER_LINEAR)
    return output
    
def predict_3d_sagcorax_stage2(model, x_inp_orig, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25):
    # x_inp_orig: (ch, h, w, d)
    # y_pred_s1: (h, w, d)
    # --- sag --- #
    x_inp = np.transpose(x_inp_orig, (1, 0, 2, 3)) # (h, ch, w, d)
    output = predict_3d_stage2(model, x_inp, y_pred_s1, img_sz=img_sz, minbbox=minbbox, bboxper=bboxper)
    y_prob = np.transpose(output, (1, 0, 2, 3))
    # --- cor --- #
    x_inp = np.transpose(x_inp_orig, (2, 0, 1, 3)) # (w, ch, h, d)
    y_pred_s1_cor = np.transpose(y_pred_s1, (1, 0, 2)) # (w, h, d)
    output = predict_3d_stage2(model, x_inp, y_pred_s1_cor, img_sz=img_sz, minbbox=minbbox, bboxper=bboxper)
    y_prob += np.transpose(output, (1, 2, 0, 3))
    # --- axial --- #
    x_inp = np.transpose(x_inp_orig, (3, 0, 1, 2)) # (d, ch, h, w)
    y_pred_s1_ax = np.transpose(y_pred_s1, (2, 0, 1)) # (d, h, w)
    output = predict_3d_stage2(model, x_inp, y_pred_s1_ax, img_sz=img_sz, minbbox=minbbox, bboxper=bboxper)
    y_prob += np.transpose(output, (1, 2, 3, 0)) 
    # -- proeb to label ---- #
    y_pred = np.squeeze(np.argmax(y_prob, 0))
    return y_pred, y_prob
    
def save_nifti(fname, dir_out, out_label):
    imgObj = nib.load(fname)
    imgObj_w = nib.Nifti1Image(out_label, imgObj.affine, imgObj.header)
    f_new = os.path.join(dir_out, fname.split('/')[-1][:-10] + '.nii.gz')
    nib.save(imgObj_w, f_new)

def refine_lable(y_pred, thrshld_lable4=500, perThrshld_blob = 10.0):
    if np.sum(y_pred==3) <=thrshld_lable4:
        y_pred[y_pred==3] = 0
    y_pred = remove_small_blobs(y_pred, perThrshld=perThrshld_blob)
    return y_pred



# ---- Arguments ---- #
ap = argparse.ArgumentParser()
ap.add_argument("-np", "--nifti_path", type=str, default='./data/nifti/val/*/*t1.nii.gz')
ap.add_argument("-op", "--out_path", type=str, default='./data/nifti/output/')
args = vars(ap.parse_args())

if __name__ == '__main__': 
    # ------- Load Model Definition -------- #
    path_to_ckpt_s1 = './weights/stage1/resnet50-scse/epoch=12-step=1442.ckpt'
    model_s1 = EncoderDecoderModule(in_chans=4, out_chans=4,  encoder_name='resnet50',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s1 = torch.load(path_to_ckpt_s1, map_location=lambda storage, loc: storage)
    model_s1.load_state_dict(checkpoint_s1['state_dict'], strict=False)
    model_s1.to(device)
    model_s1 = model_s1.eval()

    path_to_ckpt_s2_0 = './weights/stage2/Unet-resnet50-scse/epoch=33-step=3467.ckpt'
    model_s2_0 = EncoderDecoderModule(in_chans=4, out_chans=4,  encoder_name='resnet50',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_0 = torch.load(path_to_ckpt_s2_0, map_location=lambda storage, loc: storage)
    model_s2_0.load_state_dict(checkpoint_s2_0['state_dict'], strict=False)
    model_s2_0.to(device)
    model_s2_0 = model_s2_0.eval()

    path_to_ckpt_s2_1 = './weights/stage2/Unetpp-resnet50-scse/epoch=19-step=2039.ckpt'
    model_s2_1 = EncoderDecoderModuleUnetPP(in_chans=4, out_chans=4,  encoder_name='resnet50',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_1 = torch.load(path_to_ckpt_s2_1, map_location=lambda storage, loc: storage)
    model_s2_1.load_state_dict(checkpoint_s2_1['state_dict'], strict=False)
    model_s2_1.to(device)
    model_s2_1 = model_s2_1.eval()

    path_to_ckpt_s2_6 = './weights/stage2/FPN-se_resnet50/epoch=30-step=2107.ckpt'
    model_s2_2 = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='se_resnet50',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_6 = torch.load(path_to_ckpt_s2_6, map_location=lambda storage, loc: storage)
    model_s2_2.load_state_dict(checkpoint_s2_6['state_dict'], strict=False)
    model_s2_2.to(device)
    model_s2_2 = model_s2_2.eval()

    path_to_ckpt_s2_7 = './weights/stage2/FPN-dpn92/epoch=31-step=4351.ckpt'
    model_s2_3 = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='dpn92',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_7 = torch.load(path_to_ckpt_s2_7, map_location=lambda storage, loc: storage)
    model_s2_3.load_state_dict(checkpoint_s2_7['state_dict'], strict=False)
    model_s2_3.to(device)
    model_s2_3 = model_s2_3.eval()

    path_to_ckpt_s2_8 = './weights/stage2/FPN-inceptionresnetv2/epoch=29-step=4079.ckpt'
    model_s2_4 = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='inceptionresnetv2',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_8 = torch.load(path_to_ckpt_s2_8, map_location=lambda storage, loc: storage)
    model_s2_4.load_state_dict(checkpoint_s2_8['state_dict'], strict=False)
    model_s2_4.to(device)
    model_s2_4 = model_s2_4.eval()

    path_to_ckpt_s2_9 = './weights/stage2/FPN-densenet169/epoch=22-step=2345.ckpt'
    model_s2_5 = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='densenet169',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_9 = torch.load(path_to_ckpt_s2_9, map_location=lambda storage, loc: storage)
    model_s2_5.load_state_dict(checkpoint_s2_9['state_dict'], strict=False)
    model_s2_5.to(device)
    model_s2_5 = model_s2_5.eval()

    path_to_ckpt_s2_10 = './weights/stage2/FPN-efficientnet-b5/epoch=5-step=1217.ckpt'
    model_s2_6 = EncoderDecoderModuleFPN(in_chans=4, out_chans=4,  encoder_name='efficientnet-b5',  encoder_weights=None, decoder_attention_type='scse')
    checkpoint_s2_10 = torch.load(path_to_ckpt_s2_10, map_location=lambda storage, loc: storage)
    model_s2_6.load_state_dict(checkpoint_s2_10['state_dict'], strict=False)
    model_s2_6.to(device)
    model_s2_6 = model_s2_6.eval()

    # --------- Predict -------------- #
    pathVal = args['nifti_path']
    dir_out = args['out_path']
    fnames = sorted(glob.glob(pathVal))
    num_file = len(fnames)
    assert (num_file > 0), "No files in the folder"
    print('Number of files: ', num_file)
    for k, f in enumerate(fnames):
        start = time.time()
        x_inp = get_3d_images_for_validation_nogt(os.path.split(f)[0])
        # --- stage 1 ---- #
        y_pred_s1, y_prob_s1 = predict_3d_sagcorax_stage1(model_s1, x_inp)
        # ---- stage 2 ---- #
        _, y_prob_s2_0 = predict_3d_sagcorax_stage2(model_s2_0, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)  
        _, y_prob_s2_1 = predict_3d_sagcorax_stage2(model_s2_1, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        _, y_prob_s2_2 = predict_3d_sagcorax_stage2(model_s2_2, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        _, y_prob_s2_3 = predict_3d_sagcorax_stage2(model_s2_3, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        _, y_prob_s2_4 = predict_3d_sagcorax_stage2(model_s2_4, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        _, y_prob_s2_5 = predict_3d_sagcorax_stage2(model_s2_5, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        _, y_prob_s2_6 = predict_3d_sagcorax_stage2(model_s2_6, x_inp, y_pred_s1, img_sz=224, minbbox=56, bboxper=1.25)
        # ---- combine probablity --- #
        y_pred_s1s2 = np.squeeze(np.argmax(y_prob_s1+y_prob_s2_0+y_prob_s2_1+y_prob_s2_2+y_prob_s2_3+y_prob_s2_4+y_prob_s2_5+y_prob_s2_6, 0))
        # --- refine lables ---- #
        y_pred_s1s2 = refine_lable(y_pred_s1s2)
        # --- save nifiti files --- #
        y_pred_s1s2[y_pred_s1s2==3] = 4
        save_nifti(f, dir_out, y_pred_s1s2.astype('uint16'))
        # --- print time --- #
        stop = time.time()
        print(k, ' of ', len(fnames), stop-start)

