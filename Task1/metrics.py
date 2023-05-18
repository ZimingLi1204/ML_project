####根据输出的所有mask(2d)计算dice
import numpy as np
import scipy.io as scio

# mask_generated = np.load('data1/train/label/label0001.nii.gz_101.npy')
test_set = np.load('pre_processed_data1_缺部分trainset/pre_processed_dataset1_test.npz')
mask_gen = test_set["mask"]
mask_groundtruth = test_set["mask"]

info = scio.loadmat('pre_processed_data1_缺部分trainset/pre_processed_dataset1_test.mat')
cate = info["category"]
cate = cate[0, :]
CT_idx = np.int32(info["name"])
CT_idx = CT_idx.reshape(-1)
listp = [0, 0, 0, 0, 0, 0, 0]
print("data loaded")
'''
j = 1
i = 0
for i in range (2483):
    if CT_idx[i] + 1 == CT_idx[i + 1]:
        listp[j] = i
        j += 1
listp[j] = i+1
print(listp)
'''
'''def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    #intersection = 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
'''

def dice_coefficient(y_true, y_pred):
    # Flatten the arrays
    y_true_f = np.asarray(y_true).astype(np.int32)
    y_pred_f = np.asarray(y_pred).astype(np.int32)

    # Compute the intersection
    intersection = np.logical_and(y_true_f, y_pred_f)

    # Compute the Dice coefficient
    dsc = (2. * intersection.sum()) / (y_true_f.sum() + y_pred_f.sum())
    
    return dsc


def find_pointer():
    j = 1
    i = 0
    for i in range (CT_idx.shape[0] - 1):
        if CT_idx[i] + 1 == CT_idx[i + 1]:
           listp[j] = i
           j += 1
    listp[j] = i + 1

# 35 - 40 CT
def eval_mdice(case_num, gen_mask):
    '''
        case_num: 35-40 CT 
        gen_mask: N * 512 * 512
    '''
    sumdice = float(0)
    print(cate.shape)
    for i in range (1, 14):
        cate_CT = cate[listp[case_num-35]:listp[case_num-34]]
        gen_mask_CT = gen_mask[listp[case_num-35]:listp[case_num-34], :, :]
        gt_mask_CT = mask_groundtruth[listp[case_num-35]:listp[case_num-34], :, :]
        print(cate_CT.shape)
        #breakpoint()
        location_list = []
        for j, k in enumerate(cate_CT):
            if k == i:
                location_list.append(j)
        print(location_list)
        # breakpoint()
        gt_mask_cate = gt_mask_CT[location_list, :, :]
        gen_mask_cate = gen_mask_CT[location_list, :, :]
        #assert gen_mask_cate.shape[0] == gt_mask_cate.shape[0] == len(location_list)
        dice = dice_coefficient(gt_mask_cate, gen_mask_cate)
        print(i, dice)
        sumdice += dice

    mdice = sumdice / 13
    print(mdice)
    return mdice

find_pointer()
print(listp)
eval_mdice(35, mask_gen)

#dice = np.zeros(13,layer_size)

