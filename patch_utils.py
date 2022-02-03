import numpy as np

def findIndexToCropLenghtN(xmin, xmax, N, Nmax):
    '''
    Given the coordinated (xmin, xmax) of the patch find the new coordinated (xmin_new, xmax_new) such that the length (xmax_new-xmin_new)=N
    It assumes N > (xmax-xmin), desired length is greater than original length
    '''
    N_curr = xmax-xmin
    assert N>=N_curr, 'in findIndexToCropLenghtN(), desired length N should be grater that current length'
    pad = np.ceil((N-N_curr)/2.)
    xmin_new = xmin-pad if (xmin-pad)>=0 else 0
    xmax_new = xmax+pad if (xmax+pad)<Nmax else Nmax-1
    return int(xmin_new), int(xmax_new)

 
def find_bbox(img, minbbox=56, bboxper=1.25):
    '''
    Function to find the number of bounding boxes and their corrdinates
    img: label image
    minbbox: size of minimum bounding box allowed for cropping tumor region
    bboxper: size of cropping compared to bounding box 1 means same size, 1.25 means crop 25% more than the actual bounding box
    '''
    r, c = img.shape
    img_binary = np.zeros((r, c))
    img_binary[img>0] = 1
    # --- find minimum bounding box ---- #
    img_bin_sum_rows = np.sum(img_binary, 0)
    minc = min(np.argwhere(img_bin_sum_rows>0))[0]
    maxc = max(np.argwhere(img_bin_sum_rows>0))[0]
    img_bin_sum_cols = np.sum(img_binary, 1)
    minr = min(np.argwhere(img_bin_sum_cols>0))[0]
    maxr = max(np.argwhere(img_bin_sum_cols>0))[0]
    # --- find isotropic bounding box ---- #
    bboxlen = int(bboxper*max(maxr-minr, maxc-minc, minbbox))
    # --- find new coordinates ---- #
    minr, maxr = findIndexToCropLenghtN(minr, maxr, bboxlen, r)
    minc, maxc = findIndexToCropLenghtN(minc, maxc, bboxlen, c)
    bbox_list = []
    bbox_list.append([minr, minc, maxr, maxc])
    return bbox_list