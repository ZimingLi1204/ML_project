import sys

import torch
import yaml
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from data import Mydataset
import data as DT
import cv2
import numpy as np
import os
import utils.metrics as metrics
import pdb
from tqdm.auto import tqdm

def test(predictor, dataset : Mydataset):
    '''
        predictor = SamPredictor(sam) 训练好的模型
        dataset = testImage  shape = N * 512 * 512
        Return: a ndarray (N * 512 * 512)   将dataset输入模型生成的mask
    '''

    gen_mask_array = np.zeros([0, 512, 512])


    for index in tqdm(range(len(dataset)), ncols=90):

        # 这里不确定Mydataset __get_item__方法是否会保持数据集原本顺序
        img = dataset.img[index, :, :]
        gt_mask = dataset.mask[index, :, :]
        promt_type = 'box'
        promt = DT.get_promt(img, gt_mask, promt_type)

        # 加载图片
        # sam模型的输入需要img转化成3通道
        img = img.reshape(-1, 512, 512)
        img = img.repeat(3, axis=0).transpose(1, 2, 0).astype(np.uint8)
        # pdb.set_trace()
        predictor.set_image(img.astype("uint8"))

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

        # pdb.set_trace()
        # print(mask.sum())
        gen_mask_array = np.append(gen_mask_array, mask, axis=0)
        #tqdm.write(mask.sum(), np.int32(gt_mask.sum()))

    return gen_mask_array

if __name__ == "__main__":
    
    sam_checkpoint = "../pretrain_model/sam_vit_h.pth"####!!!!!!!一定要把pretrain model名字改了!!!!!!
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("model loaded_______________________________________________________")

    cfg_file = open('cfg.yaml')
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)
    cfg_file.close()

    info_test_path = "../BTCV_dataset1/pre_processed_dataset1_test" ####!!!!!!!一定要把文件夹名字改了!!!!!!
    if cfg['data']["use_embedded"]:
        data_test_path = "../BTCV_dataset1/vit-h_embedding_bc1_test" 
    else:
        data_test_path = info_test_path

    dataset = DT.load_test_data_from_dir(info_test_path = info_test_path, data_test_path=data_test_path, cfg=cfg)

    print("dataset loaded______________________________________________________")

    gen_mask = test(predictor=predictor, dataset=dataset)
    pdb.set_trace()
    if not os.path.exists("result"):
        os.mkdir("result")
    save_path = "result/prompt_1"
    np.savez_compressed(save_path, mask=gen_mask)
    print("result saved in ({}) _____________________".format(save_path))



    '''
    Evaluation Metrics:
    First, adjust the filepath in metrics.py
    Second, change iter to the number of CT cases
    At last, change the generated mask variable
    '''
    
    # gen_mask = np.zeros([2205, 512, 512])

    dice = metrics.Dice()

    dice.eval_data_processing(iter = 6, gen_mask = gen_mask)
