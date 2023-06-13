####根据输出的所有mask(2d)计算dice
import numpy as np
import scipy.io as scio
import pdb

'''
Before you use: 
Adjust the Filepath to the reference set HERE!!!!!!
'''

def dice_coefficient(y_true, y_pred):
    # Flatten the arrays
    # np.seterr(divide='ignore',invalid='ignore')
    y_true_f = np.asarray(y_true).astype(np.int32)
    y_pred_f = np.asarray(y_pred).astype(np.int32)
    # print(y_pred_f.shape, y_true_f.shape)
    # Compute the intersection
    intersection = np.logical_and(y_true_f, y_pred_f)
    #y_pred_f = np.logical_and(y_pred_f, y_pred_f)
    # if intersection.sum() == 0:
    #     dsc = 0
    # # Compute the Dice coefficient
    # else :
    idx = (y_true_f.sum(axis=(-1, -2)) != 0)

    intersection = intersection[idx]
    y_pred_f = y_pred_f[idx]
    y_true_f = y_true_f[idx]

    dsc = (2. * intersection.sum(axis=(-1, -2))) / (y_true_f.sum(axis=(-1, -2)) + y_pred_f.sum(axis=(-1, -2)) + 1e-12)

    # try:
    #     if(dsc.min() == 0):
    #         pdb.set_trace()
    # except:
    #     pdb.set_trace()
        #print(intersection.sum())
        #print(y_true_f.sum(), y_pred_f.sum())
    
    return dsc

class Dice():
    def __init__(self, data_path = '../BTCV_dataset1/pre_processed_dataset1_test.npz',
                  info_path = '../BTCV_dataset1/pre_processed_dataset1_test.mat', mask_gt = None, info_gt = None, verbose=False) -> None:
        '''
        如果给了数据(mask)和info, 那么就不再需要读取
        '''

        self.verbose = verbose

        if (mask_gt is None):
            test_set = np.load(data_path)
            self.mask_groundtruth = test_set["mask"]
        else:
            self.mask_groundtruth = mask_gt
        if info_gt is None:
            info = scio.loadmat(info_path)
        else:
            info = info_gt

        fla = 0
        for i in range(self.mask_groundtruth.shape[0]):
            if np.max(self.mask_groundtruth[i, :, :]) != 1:
                print(i)
                fla = 1
        if fla != 0:
            print("test data is not correct")
        else :
            print("test data is correct")        

        self.cate = info["category"]
        self.cate = self.cate[0, :]
        self.CT_idx = np.int32(info["name"])
        self.CT_idx = self.CT_idx.reshape(-1)
        self.listp = [0]
        print("data loaded")
        # pdb.set_trace()
        self.find_pointer()

    def find_pointer(self):
        # j = 1
        # i = 0
        for i in range (self.CT_idx.shape[0] - 1):
            if self.CT_idx[i] + 1 == self.CT_idx[i + 1]:
                self.listp.append(i+1)
        self.listp.append(self.CT_idx.shape[0])
        # pdb.set_trace()

    # 35 - 40 CT
    def eval_mdice(self, case_num, gen_mask):
        '''
            case_num: 35-40 CT 
            gen_mask: N * 512 * 512
        '''
        sumdice = float(0)
        #print(cate.shape)
        mdice = [0] * 13
        organ_num = 0
        cate_CT = self.cate[self.listp[case_num]:self.listp[case_num+1]]
        gen_mask_CT = gen_mask[self.listp[case_num]:self.listp[case_num+1], :, :]
        gt_mask_CT = self.mask_groundtruth[self.listp[case_num]:self.listp[case_num+1], :, :]

        for i in range (1, 14):
            #print(cate_CT.shape)
            #breakpoint()
            # location_list = []
            # for j, k in enumerate(cate_CT):
            #     if k == i:
            #         location_list.append(j)
            # pdb.set_trace()
            location_list = np.where(cate_CT == i)
            
            #print(location_list)
            # breakpoint()
            gt_mask_cate = gt_mask_CT[location_list, :, :]
            gen_mask_cate = gen_mask_CT[location_list, :, :]
            # pdb.set_trace()
            #assert gen_mask_cate.shape[0] == gt_mask_cate.shape[0] == len(location_list)
            if (gt_mask_cate.shape[0] != 0):
                organ_num += 1
                dice = dice_coefficient(gt_mask_cate, gen_mask_cate)
                if self.verbose:
                    print("Organ:", i, "Dice:", dice)
                # sumdice += dice
                mdice[i-1] = dice

        if self.verbose:
            print("mDice", mdice)
        # pdb.set_trace()
        return mdice

    def eval_data_processing(self, iter, gen_mask):
        #寻找每一个CT对应的编号
        if self.verbose:
            print("list_p:", self.listp)
        m_Dice = [[] for i in range(13)]
        for i in range (iter):
            if self.verbose:
                print("CT", i + 1, "_______________________________")
            m_Dice_new = self.eval_mdice(i, gen_mask)
            # pdb.set_trace()
            for j, d in enumerate(m_Dice_new):
                m_Dice[j].append(d)
        
        # pdb.set_trace()

        for i, d in enumerate(m_Dice):
            m_Dice[i] = np.hstack(d).mean()

        # print("Total mDice:", m_Dice)
        return m_Dice
        #六个CT对应的m_Dice数据   

