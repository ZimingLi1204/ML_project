## pre-process data and label from BTCV/data1/train (30 CT)

import numpy as np
import os
import re
import random
import scipy.io as scio
import datetime

def get_mask_category(label : np.ndarray):
    '''
        对一个label数据 512 * 512, 计算其包含了哪些类别
        Retrun: a list 记录label中包含的所有类别(1-13)
    '''
    mask_category_list = []
    for i in range(1, 14):
        if i in label:
            mask_category_list.append(i)
    return mask_category_list

if __name__ == "__main__":

    # 将30个CT安排为train val 和 test
    # train 编号为 1-10 21-28
    # val 编号为 29-34
    # test 编号为 35-40

    # 18个CT用于train
    train_img = np.zeros([0, 512, 512])  # ndarray   N * 512 * 512
    train_mask = np.zeros([0, 512, 512])   # ndarray N * 512 * 512
    train_name = []  # a list  len = N  将数据映射到30个编号
    train_slice_id = [] # a list len = N 将数据映射到切片编号
    train_category = [] # a list len = N 将0-1mask映射到13个类别

    # 6个CT用于validate
    val_img = np.zeros([0, 512, 512])
    val_mask = np.zeros([0, 512, 512])
    val_name = []
    val_slice_id = []
    val_category = []

    # 6个CT用于test
    test_img = np.zeros([0, 512, 512])
    test_mask = np.zeros([0, 512, 512])
    test_name = []
    test_slice_id = []
    test_category = []


    img_dir = "BTCV/data1/train/img"
    label_dir = "BTCV/data1/train/label"
    img_path_list = os.listdir(img_dir)

    # 记录每条数据（2d图片+2d mask）的编号(name)以及 是第几个切片(slice_id)
    name = []
    slice_id = []

    for img_path in img_path_list:
        # 2d图片命名为 img\d{4}.nii.gz_\d{1,}.npy
        name.append(re.search(pattern="\\d{4}", string=img_path).group(0)) # 00xx : str
        slice_id.append(re.search(pattern="_(\\d{1,})", string=img_path).group(1)) # xxx : str

    print(datetime.datetime.now())
    # 生成train val test数据集 N * 512 * 512 N为0-1mask数量，每张2d图片最多对应13个0-1mask
    for index in range(len(slice_id)):
        img_path = img_dir + '/img' + name[index] + '.nii.gz_' + slice_id[index] + '.npy'
        label_path = label_dir + '/label' + name[index] + '.nii.gz_' + slice_id[index] + '.npy'
        img = np.load(img_path)
        label = np.load(label_path)

        mask_category_list = get_mask_category(label)

        if 1 <= int(name[index]) <= 28:   # train dataset
            for mask_category in mask_category_list:
                train_img = np.concatenate((train_img, img.reshape(-1, img.shape[0], img.shape[1])), axis=0)
                zero_one_mask = np.array([x == mask_category for x in label])
                train_mask = np.concatenate((train_mask, zero_one_mask.reshape(-1, zero_one_mask.shape[0], zero_one_mask.shape[1])), axis=0)
                train_name.append(name[index])
                train_slice_id.append(slice_id[index])
                train_category.append(mask_category)
        elif 35 <= int(name[index]) <= 40:    # test dataset
            for mask_category in mask_category_list:
                test_img = np.concatenate((test_img, img.reshape(-1, img.shape[0], img.shape[1])), axis=0)
                zero_one_mask = np.array([x == mask_category for x in label])
                test_mask = np.concatenate((test_mask, zero_one_mask.reshape(-1, zero_one_mask.shape[0], zero_one_mask.shape[1])), axis=0)
                test_name.append(name[index])
                test_slice_id.append(slice_id[index])
                test_category.append(mask_category)
        elif 29 <= int(name[index]) <= 34:    # validate dataset
            for mask_category in mask_category_list:
                val_img = np.concatenate((val_img, img.reshape(-1, img.shape[0], img.shape[1])), axis=0)
                zero_one_mask = np.array([x == mask_category for x in label])
                val_mask = np.concatenate((val_mask, zero_one_mask.reshape(-1, zero_one_mask.shape[0], zero_one_mask.shape[1])), axis=0)
                val_name.append(name[index])
                val_slice_id.append(slice_id[index])
                val_category.append(mask_category)
        else:
            raise Exception

        if index % 100 == 0:
            print("current index: ", index, "_________________________")

# save val dataset
np.savez_compressed("BTCV/pre_processed_dataset1_val", img=val_img,mask=val_mask)
scio.savemat("BTCV/pre_processed_dataset1_val.mat", mdict={"name":val_name, "slice_id":val_slice_id, "category":val_category})


# save test dataset
np.savez_compressed("BTCV/pre_processed_dataset1_test", img=test_img,mask=test_mask)
scio.savemat("BTCV/pre_processed_dataset1_test.mat", mdict={"name":test_name, "slice_id":test_slice_id, "category":test_category})

# save train dataset
np.savez_compressed("BTCV/pre_processed_dataset1_train", img=train_img,mask=train_mask)
scio.savemat("BTCV/pre_processed_dataset1_train.mat", mdict={"name":train_name, "slice_id":train_slice_id, "category":train_category})

print(datetime.datetime.now())
# # 保存数据集为.mat格式 可以用scipy读取
# scio.savemat(file_name="BTCV/pre_processed_dataset1.mat", mdict={
#     "train":{"img":train_img, "mask":train_mask, "name":train_name, "slice_id":train_slice_id, "category":train_category},
#     "test":{"img":test_img, "mask":test_mask, "name":test_name, "slice_id":test_slice_id, "category":test_category},
#     "validate":{"img":val_img, "mask":val_mask, "name":val_name, "slice_id":val_slice_id, "category":val_category}})
