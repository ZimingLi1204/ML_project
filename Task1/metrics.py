####根据输出的所有mask(2d)计算dice
import numpy as np
import scipy.io as scio

mask_threshold = 2
mask_offset = 1
data_size = np.int32(1)
layer_size = 12

mask_generated = np.load('data1/train/label/label0001.nii.gz_101.npy')
test_set = np.load('pre_processed_data1_缺部分trainset/pre_processed_dataset1_test.npz')
mask_groundtruth = test_set["mask"]

info = scio.loadmat('pre_processed_data1_缺部分trainset/pre_processed_dataset1_test.mat')
cate = info["category"]
slice = np.int32(info["slice_id"])
CT_idx = np.int32(info["name"])
CT_idx = CT_idx.reshape(2484)
case_num = 29
listp = [0, 466, 949, 1431, 1877, 2190, 2483, 0]

#j = 1
#i = 0
#for i in range (2483):
#    if CT_idx[i] + 1 == CT_idx[i + 1]:
#        listp[j] = i
#        j += 1
#listp[j] = i+1
print(listp)

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    #intersection = 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def dice_coefficient(y_true, y_pred):
    # Flatten the arrays
    y_true_f = np.asarray(y_true).astype(np.int32)
    y_pred_f = np.asarray(y_pred).astype(np.int32)

    # Compute the intersection
    intersection = np.logical_and(y_true_f, y_pred_f)

    if intersection.sum() == 0:
        return 0
    # Compute the Dice coefficient
    dsc = (2. * intersection.sum()) / (y_true_f.sum() + y_pred_f.sum())
    
    return dsc



sumdice = float(0)
for i in range (1, 14):
    organlist = np.where(cate[listp[case_num-29]:listp[case_num-28]] == i)
    mask_organ = mask_groundtruth[organlist, :, :]
    dice = dice_coefficient(mask_organ, mask_generated)
    print(i, dice)
    sumdice += dice

mdice = sumdice / 13
print(mdice)
#dice = np.zeros(13,layer_size)

