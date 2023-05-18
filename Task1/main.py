import sys

import torch

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from Task2.data import Mydataset
import Task2.data as DT
import cv2
import numpy as np
import os



def test(predictor, dataset : Mydataset):
    '''
        predictor = SamPredictor(sam) 训练好的模型
        dataset = testImage  shape = N * 512 * 512
        Return: a ndarray (N * 512 * 512)   将dataset输入模型生成的mask
    '''

    gen_mask_array = np.zeros([0, 512, 512])


    for index in range(len(dataset)):

        # 这里不确定Mydataset __get_item__方法是否会保持数据集原本顺序
        img = dataset.img[index, :, :]
        gt_mask = dataset.mask[index, :, :]
        promt_type = dataset.promt_type
        promt = DT.get_promt(img, gt_mask, promt_type)

        # 加载图片
        # sam模型的输入需要img转化成3通道
        img = img.reshape(-1, 512, 512)
        img = img.repeat(3, axis=0).transpose(1, 2, 0)
        predictor.set_image(img)

        '''The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.'''
        ''' point_coords: A Nx2 array. Each point is in (X,Y) in pixels.
            point_labels: A length N array of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.'''
        if promt_type == 'single_point':
            promt_coords, promt_label = promt
            mask, _, _ = predictor.predict(promt_coords, promt_label, multimask_output=False)
            ##  在单点情况下multimask_output也可以设置为 True，再选取confidence_score最高的mask
        elif promt_type == 'points':
            promt_coords, promt_label = promt
            mask, _, _ = predictor.predict(promt_coords, promt_label, multimask_output=False)

        # box: A length 4 array , in XYXY format.
        elif promt_type == 'box':
            mask, _, _ = predictor.predict(box = promt, multimask_output=False)

        # mask_input: 1*256*256
        elif promt_type == 'mask':
            mask, _, _ = predictor.predict(mask_input = promt, multimask_output=False)
        else:
            raise Exception

        # multimask_output设为 False, 对每个img只输出1个mask
        assert mask.shape == (1, 512, 512)

        gen_mask_array = np.append(gen_mask_array, mask, axis=0)
        print(gen_mask_array.shape)

    return gen_mask_array

if __name__ == "__main__":
    sam_checkpoint = "pretrain_model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("model loaded_______________________________________________________")

    data_test_path = "BTCV/pre_processed_dataset1_test"
    dataset = DT.load_test_data_from_dir(data_test_path=data_test_path)
    print("dataset loaded______________________________________________________")

    gen_mask = test(predictor=predictor, dataset=dataset)

    if not os.path.exists("./result"):
        os.mkdir("result")
    save_path = "result/mask_generated_from_testset"
    np.savez_compressed(save_path, mask=gen_mask)
    print("result saved in ({}) _____________________".format(save_path))

    ###TODO###
    # evaluation metrics
    # return eval_mDice(gen_mask, gt_mask)