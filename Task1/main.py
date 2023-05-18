import sys
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

    for img, gt_mask, promt, promt_type in dataset:
        # 加载图片   相同的图片可以不重复加载，提升代码运行速度
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

        gen_mask_array = np.append(gen_mask_array, mask.reshape(-1, mask.shape[0], mask.shape[1]), axis=0)
        print(gen_mask_array.shape)

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

    ###TODO###
    # evaluation metrics
    # return eval_mDice(mask, gt_mask)
