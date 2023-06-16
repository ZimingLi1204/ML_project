import torch
import numpy as np
import os
import random
import cv2
import pdb


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 2333
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)  # random seed for random module
    np.random.seed(seed)  # for np module
    torch.manual_seed(seed)  # for pytorch module
    os.environ["PYTHONHASHSEED"] = str(seed)  # for os env Python hash seed
    torch.cuda.manual_seed(seed)  # cuda manual seed
    torch.cuda.manual_seed_all(seed)  # cuda manual seed all


def get_promt(
    mask,
    promt_type="single_point",
    point_num=16,  # points 选取的点数
    mask_per=0.5,  # points 选取的1-label点比例
    center_point=True,
    # 选点模式，False 为直接随机选点
    # True 为先随机选出point_size个点(coord_size)再选出最中心的点
    point_size=16,
):
    ###TODO###
    # 根据输入img和mask生成promt
    # box or mask or points or single_point!!!
    # 需要保证生成的 point promt均在mask前景中
    # 不同类型promt 的具体格式见 segment_anything/predictor.py 104-130行注释

    promt = None

    mask = mask.astype(np.uint8)

    one_point_num = int(point_num * mask_per)  # points 选取的0-label点数
    zero_point_num = point_num - one_point_num  # points 选取的1-label点数

    if promt_type == "single_point":  # 单点 1个XY坐标 和 1个01 label
        if center_point:
            coord_set = np.zeros((point_size, 2), dtype=np.int32)
            coor_1d = np.random.choice(
                mask.shape[0] * mask.shape[1],
                size=point_size,
                p=mask.reshape(-1) / mask.sum(),
                replace=mask.sum() < point_size,
            )
            coord_set[:, 0] = coor_1d // mask.shape[0]
            coord_set[:, 1] = coor_1d % mask.shape[0]
            # 随机选point_size个点
            avg = np.mean(coord_set.astype(np.float32), axis=0).astype(np.int32)
            argmin = np.argmin(np.sum((coord_set - avg) ** 2, axis=1))
            # 选中心点
            coord = np.array([[coord_set[argmin, 0], coord_set[argmin, 1]]])
            label = np.array([1])
            promt = coord, label
        else:
            # 随机取一个在mask前景中的XY坐标
            coord = np.zeros((1, 2), dtype=np.int32)
            coor_1d = np.random.choice(
                mask.shape[0] * mask.shape[1],
                size=1,
                p=mask.reshape(-1) / mask.sum(),
                replace=mask.sum() < 1,
            )
            coord[:, 0] = coor_1d // mask.shape[0]
            coord[:, 1] = coor_1d % mask.shape[0]
            label = np.array([1])
            promt = coord, label

    elif promt_type == "points":  # 多点   N个XY坐标 和 N个01 label
        if center_point:
            coord = np.empty((0, 2), dtype=np.int32)
            for m in range(one_point_num):
                coord_set = np.zeros((point_size, 2), dtype=np.int32)
                coor_1d = np.random.choice(
                    mask.shape[0] * mask.shape[1],
                    size=point_size,
                    p=mask.reshape(-1) / mask.sum(),
                    replace=mask.sum() < point_size,
                )
                coord_set[:, 0] = coor_1d // mask.shape[0]
                coord_set[:, 1] = coor_1d % mask.shape[0]
                # 随机选point_size个1 label点
                avg = np.mean(coord_set.astype(np.float32), axis=0).astype(np.int32)
                argmin = np.argmin(np.sum((coord_set - avg) ** 2, axis=1))
                # 选中心点
                coord = np.concatenate(
                    (coord, [[coord_set[argmin, 0], coord_set[argmin, 1]]])
                )
            for m in range(zero_point_num):
                coord_set = np.zeros((point_size, 2), dtype=np.int32)
                coor_1d = np.random.choice(
                    mask.shape[0] * mask.shape[1],
                    size=point_size,
                    p=(1 - mask.reshape(-1)) / ((1 - mask).sum()),
                    replace=(1 - mask).sum() < point_size,
                )
                coord_set[:, 0] = coor_1d // mask.shape[0]
                coord_set[:, 1] = coor_1d % mask.shape[0]
                # 随机选point_size个0 label点
                avg = np.mean(coord_set.astype(np.float32), axis=0).astype(np.int32)
                argmin = np.argmin(np.sum((coord_set - avg) ** 2, axis=1))
                # 选中心点
                coord = np.concatenate(
                    (coord, [[coord_set[argmin, 0], coord_set[argmin, 1]]])
                )
            # label = np.array([mask[coord[i][0]][coord[i][1]] for i in range(point_num)])
            label = mask[coord[:, 0], coord[:, 1]]
            promt = coord, label
        else:
            coord = np.zeros((point_num, 2), dtype=np.int32)
            coor_1d_1 = np.random.choice(
                mask.shape[0] * mask.shape[1],
                size=one_point_num,
                p=mask.reshape(-1) / mask.sum(),
                replace=mask.sum() < one_point_num,
            )
            coor_1d_0 = np.random.choice(
                mask.shape[0] * mask.shape[1],
                size=zero_point_num,
                p=(1 - mask.reshape(-1)) / (1 - mask).sum(),
                replace=(1 - mask).sum() < zero_point_num,
            )
            coor_1d = np.concatenate((coor_1d_0, coor_1d_1), axis=0)
            coord[:, 0] = coor_1d // mask.shape[0]
            coord[:, 1] = coor_1d % mask.shape[0]
            # label = np.array(
            #     [mask[coord[i][0]][coord[i][1]] for i in range(one_point_num)]
            # )
            # pdb.set_trace()
            label = mask[coord[:, 0], coord[:, 1]]
            promt = coord, label
            
    elif promt_type == "grid_points":  # 格点 + 中心点

        step = mask.shape[0] // point_num
        index_mask = np.zeros_like(mask)
        index_mask[np.arange(mask.shape[0]) % step == 0] = 1
        index_mask[0] = 0
        index_mask[:, np.arange(mask.shape[1]) % step != 0] = 0

        coord = np.stack(np.nonzero(index_mask)).T
        label = mask[coord[:, 0], coord[:, 1]]

        
        coord_set = np.zeros((point_size, 2), dtype=np.int32)
        coor_1d = np.random.choice(
            mask.shape[0] * mask.shape[1],
            size=point_size,
            p=mask.reshape(-1) / mask.sum(),
            replace=mask.sum() < point_size,
        )
        coord_set[:, 0] = coor_1d // mask.shape[0]
        coord_set[:, 1] = coor_1d % mask.shape[0]
        # 随机选point_size个点
        avg = np.mean(coord_set.astype(np.float32), axis=0).astype(np.int32)
        argmin = np.argmin(np.sum((coord_set - avg) ** 2, axis=1))
        # 选中心点
        coord = np.concatenate((coord, np.array([[coord_set[argmin, 0], coord_set[argmin, 1]]])), axis = 0)
        label = np.concatenate((label, np.array([1])), axis = 0)
        # pdb.set_trace()
        promt = coord, label

    elif promt_type == "box":  # 边界框  形如XYXY
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=4
        )
        # print(stats)
        coord = np.array(
            [
                stats[1][0],
                stats[1][1],
                stats[1][0] + stats[1][2],
                stats[1][1] + stats[1][3],
            ]
        )
        promt = coord
    elif promt_type == "mask":  # mask类型prompt
        pass
    else:
        raise Exception

    return promt


###used for debug
if __name__ == "__main__":
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:220, 210:280] = 1

    get_promt(
        mask=mask,
        promt_type="grid_points",
        center_point=False,
    )
