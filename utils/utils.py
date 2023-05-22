import torch
import numpy as np
import os
import random
import cv2

def set_seed(seed, torch_deterministic = False):
    if seed == -1 and torch_deterministic:
        seed = 2333
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed) # random seed for random module
    np.random.seed(seed) # for np module
    torch.manual_seed(seed) # for pytorch module
    os.environ['PYTHONHASHSEED'] = str(seed) # for os env Python hash seed
    torch.cuda.manual_seed(seed) # cuda manual seed
    torch.cuda.manual_seed_all(seed) # cuda manual seed all

def get_promt(img, mask, promt_type = "single_point", point_num = 1, box_num = 1):
    ###TODO###
    #根据输入img和mask生成promt
    # box or mask or points or single_point!!!
    # 需要保证生成的 point promt均在mask前景中
    # 不同类型promt 的具体格式见 segment_anything/predictor.py 104-130行注释

    promt = None

    if promt_type == "single_point":   # 单点 1个XY坐标 和 1个01 label
        coord = np.random.randint(low=1, high=512, size=(1, 2))
        while mask[coord[0, 0], coord[0, 1]] == 0:      # 随机取一个在mask前景中的XY坐标
            coord = np.random.randint(low=1, high=512, size=(1, 2))
        label = np.array([mask[coord[0, 0], coord[0, 1]]])
        promt = coord, label
    elif promt_type == "points":   # 多点   N个XY坐标 和 N个01 label
        coord = np.random.randint(low=1, high=512, size=(point_num, 2))
        label = np.array([mask[coord[0, 0], coord[0, 1]] for i in range(point_num)])
        promt = coord, label
    elif promt_type == "box":   # 边界框  形如XYXY
        mask=mask.astype( np.uint8 )
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        #print(stats)
        coord = np.array([stats[1][0], stats[1][1], stats[1][0]+stats[1][2], stats[1][1]+stats[1][3]])
        promt = coord
    elif promt_type == "mask":   # mask类型prompt
        pass
    else:
        raise Exception

    return promt
