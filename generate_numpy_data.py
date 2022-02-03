import glob
import os
import numpy as np
import nibabel as nb
import argparse

def get_dir_list(train_path):
    fnames = glob.glob(train_path)
    list_train = []
    for k, f in enumerate(fnames):
            list_train.append(os.path.split(f)[0])
    return list_train

def ParseData(list_data):
    '''
    Creates a list of all the slices 
    '''
    data_instance = []
    for dir_name in list_data:
        fname = glob.glob(os.path.join(dir_name, '*seg.nii.gz'))
        f = nb.load(fname[0])
        img = f.get_fdata().astype('float32')
        h, w, d = f.shape # sag, cor, ax
        for slc in range(h):
            if np.sum(img[slc, :, :]) != 0:
                data_instance.append([dir_name, 'sag', slc])
        for slc in range(w):
            if np.sum(img[:, slc, :]) != 0:
                data_instance.append([dir_name, 'cor', slc])
        for slc in range(d):
            if np.sum(img[:, :, slc]) != 0:
                data_instance.append([dir_name, 'ax', slc])
    print('Number of images: ', len(data_instance))
    return data_instance


def get_slice(dir_name, orient, slc, cont, isNorm=True):
    '''
    takes the directory name, orientation, slice number and reads a slice, zero pad/crop and normalize 
    '''
    # ---- get slice for given contrast image ---- #
    fname = glob.glob(os.path.join(dir_name, cont))
    f = nb.load(fname[0])
    img = np.squeeze(f.get_fdata()).astype('float32')
    if orient == 'sag':
        x = img[slc, :, :]
    elif orient == 'cor':
        x = img[:, slc, :]
    else:
        x = img[:, :, slc]
    return np.expand_dims(x, 0)

def get_batchsize_one(dir_name, orient, slc):
    '''
    takes index and generates one sample of input data 
    '''
    # ---- get images ---- #
    x_t1 = get_slice(dir_name, orient, slc, '*flair.nii.gz')
    x_t2 = get_slice(dir_name, orient, slc, '*t1.nii.gz')
    x_t1ce = get_slice(dir_name, orient, slc, '*t2.nii.gz')
    x_flair = get_slice(dir_name, orient, slc, '*t1ce.nii.gz')
    x_seg = get_slice(dir_name, orient, slc, '*seg.nii.gz', isNorm=False).astype('int')
    x_seg[x_seg==4] = 3
    x_inp = np.concatenate((x_t1, x_t2, x_t1ce, x_flair, x_seg), 0)
    # (flair, t1, t2, t1ce)
    return x_inp
        
def generate_data(src_path, dst_path):
    data_instance = ParseData(get_dir_list(src_path))
    for k, data in enumerate(data_instance):
        print(k, ' of ', len(data_instance))
        dir_name, orient, slc = data[0], data[1], data[2]
        x_inp = get_batchsize_one(dir_name, orient, slc)
        fname = os.path.join(dst_path, str(k)+'.npy')
        np.save(fname, x_inp)

# ---- Arguments ---- #
ap = argparse.ArgumentParser()
ap.add_argument("-sp", "--src_path", type=str, default='./data/nifti/train/*/*seg.nii.gz')
ap.add_argument("-dp", "--dst_path", type=str, default='./data/np/train/')
args = vars(ap.parse_args())
       
if __name__ == '__main__':
    '''
    Script to convert nifti images to numpy array for faster loading
    '''
    src_path = args['src_path']
    dst_path = args['dst_path']
    generate_data(src_path, dst_path)
    