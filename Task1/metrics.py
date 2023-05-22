####根据输出的所有mask(2d)计算dice
import numpy as np
import scipy.io as scio


'''
Before you use: 
Adjust the Filepath to the reference set HERE!!!!!!
'''
test_set = np.load('../BTCV/pre_processed_dataset1_test.npz')
info = scio.loadmat('../BTCV/pre_processed_dataset1_test.mat')



mask_groundtruth = test_set["mask"]
fla = 0
for i in range(mask_groundtruth.shape[0]):
    if np.max(mask_groundtruth[i, :, :]) != 1:
        print(i)
        fla = 1
if fla != 0:
    print("test data is not correct")
else :
    print("test data is correct")

cate = info["category"]
cate = cate[0, :]
CT_idx = np.int32(info["name"])
CT_idx = CT_idx.reshape(-1)
listp = [0] * 30
print("data loaded")


def dice_coefficient(y_true, y_pred):
    # Flatten the arrays
    np.seterr(divide='ignore',invalid='ignore')
    y_true_f = np.asarray(y_true).astype(np.int32)
    y_pred_f = np.asarray(y_pred).astype(np.int32)
    print(y_pred_f.shape)
    # Compute the intersection
    intersection = np.logical_and(y_true_f, y_pred_f)
    #y_pred_f = np.logical_and(y_pred_f, y_pred_f)
    if intersection.sum() == 0:
        dsc = 0
    # Compute the Dice coefficient
    else :
        dsc = (2. * intersection.sum()) / (y_true_f.sum() + y_pred_f.sum())
        #print(intersection.sum())
        #print(y_true_f.sum(), y_pred_f.sum())
    
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
    #print(cate.shape)
    for i in range (1, 14):
        cate_CT = cate[listp[case_num]:listp[case_num+1]]
        gen_mask_CT = gen_mask[listp[case_num]:listp[case_num+1], :, :]
        gt_mask_CT = mask_groundtruth[listp[case_num]:listp[case_num+1], :, :]
        #print(cate_CT.shape)
        #breakpoint()
        location_list = []
        for j, k in enumerate(cate_CT):
            if k == i:
                location_list.append(j)
        #print(location_list)
        # breakpoint()
        gt_mask_cate = gt_mask_CT[location_list, :, :]
        gen_mask_cate = gen_mask_CT[location_list, :, :]
        #assert gen_mask_cate.shape[0] == gt_mask_cate.shape[0] == len(location_list)
        dice = dice_coefficient(gt_mask_cate, gen_mask_cate)
        print("Organ:", i, "Dice:", dice)
        sumdice += dice

    mdice = sumdice / 13
    print("mDice", mdice)
    return mdice

def eval_data_processing(iter, gen_mask):
    find_pointer()
    #寻找每一个CT对应的编号
    print(listp)
    m_Dice = [0] * iter
    for i in range (iter):
        print("CT", i + 1, "_______________________________")
        m_Dice[i] = eval_mdice(i, gen_mask)
    print("Total mDice:", m_Dice)
    #六个CT对应的m_Dice数据   
