import numpy as np
import imageio
import pdb
import os

def drawpoint(img: np.ndarray, x: int, y: int, mask=None):
    if mask is None: # box
        for i in range(0, 3):
            for j in range(0, 3):
                img[x + i - 1][y + j - 1][1] = 255
                img[x + i - 1][y + j - 1][0] = 0
                img[x + i - 1][y + j - 1][2] = 0
    else:
        for i in range(0, 3):
            for j in range(0, 3):
                img[x + i - 1][y + j - 1][1] = 255 * (1-mask[x][y])
                img[x + i - 1][y + j - 1][0] = 255 * mask[x][y]
                img[x + i - 1][y + j - 1][2] = 0


class visualize():

    def __init__(self, promt_type) -> None:
        test_set = np.load("/BTCV_testset/pre_processed_dataset1_test.npz")
        self.test_img = test_set["img"]
        # self.visualize_list = [i * 100 for i in range(23)]
        self.visualize_list = [i * 100 for i in range(23)]
        self.data_root = '../img/' + promt_type
        if not os.path.exists(self.data_root):
            os.mkdir(self.data_root)

    def mask_visualize(
        self,
        gt_mask: np.ndarray,
        gen_mask: np.ndarray,
        promt: np.ndarray,
        promt_type: str,
        img_number: int,
    ):
        if img_number not in self.visualize_list:
            return
        img = self.test_img[img_number]

        if promt_type == "box":
            promt_img = np.moveaxis([img, img, img], 0, 2)
            if promt.shape == (4,):
                for x in range(promt[1], promt[3] + 1):
                    drawpoint(promt_img, x, promt[0])
                    drawpoint(promt_img, x, promt[2])
                for y in range(promt[0], promt[2] + 1):
                    drawpoint(promt_img, promt[1], y)
                    drawpoint(promt_img, promt[3], y)
            else:
                raise AssertionError
        elif promt_type == "single_point":
            promt_img = np.moveaxis([img, img, img], 0, 2)
            # pdb.set_trace()
            if promt.ndim == 1:
                x = promt[0]
                y = promt[1]
                drawpoint(promt_img, x, y, gt_mask)
            else:
                raise AssertionError
            
        elif promt_type == "points" or promt_type == "grid_points":
            promt_img = np.moveaxis([img, img, img], 0, 2)
            if promt.ndim == 2:
                for p in range(promt.shape[0]):
                    x = promt[p][0]
                    y = promt[p][1]
                    drawpoint(promt_img, x, y, gt_mask)
            else:
                raise AssertionError
        else:
            promt_img = np.moveaxis([img, img, img], 0, 2)
        
        mask_img = np.moveaxis(
            [
                img + (255 - img) * gen_mask,
                img * (1 - gen_mask),
                img * (1 - gen_mask),
            ],
            0,
            2,
        )
        gt_img = np.moveaxis(
            [
                img * (1 - gt_mask),
                img * (1 - gt_mask),
                img + (255 - img) * gt_mask,
            ],
            0,
            2,
        )
        d = img.shape[0]
        s = 20
        img_base = np.ones((d, d * 4 + s * 3, 3)) * 255
        # pdb.set_trace()
        img_base[:, :d] = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img_base[:, d+s:2*d+s] = gt_img
        img_base[:, 2*d+2*s:3*d+2*s] = promt_img
        img_base[:, 3*d+3*s:4*d+3*s] = mask_img
        # img_output = np.concatenate((promt_img, mask_img, gt_img)).astype(np.uint8)
        if not os.path.exists(self.data_root + '/' + str(img_number)):
            os.mkdir(self.data_root + '/' + str(img_number))
        imageio.imwrite(
            self.data_root + '/' + str(img_number) +  "/raw.png",
            img.astype(np.uint8)
        )
        imageio.imwrite(
            self.data_root + '/' + str(img_number) +  "/promt.png",
            promt_img.astype(np.uint8)
        )
        imageio.imwrite(
            self.data_root + '/' + str(img_number) +  "/mask.png",
            mask_img.astype(np.uint8)
        )
        # pdb.set_trace()
        imageio.imwrite(
            self.data_root + '/' + str(img_number) +  "/gt.png",
            gt_img.astype(np.uint8)
        )

        imageio.imwrite(
            self.data_root + '/' + str(img_number) +  "/all.png",
            img_base.astype(np.uint8)
        )


if __name__ == '__main__':
    root = "/root/autodl-tmp/ML_project/img"
    promt = ['box', 'grid_points_24', 'grid_points_16', 'grid_points_12', 'points', 'points_center', 'single_point_center', 'single_point']
    l = len(promt)
    d = 512
    s = 20
    from tqdm import tqdm
    for j in tqdm([j*100 for j in range(23)], ncols=90):
        imgs = np.ones((d * l + s * (l-1), d * 4 + s * 3, 3)) * 255
        for i, p in enumerate(promt):
            path = os.path.join(root, p, str(j), 'all.png')
            imgs[(i) * d + (i) * s: (i+1)*d + (i) * s] = imageio.v2.imread(path)

        imageio.imwrite(
                root + "/all_{}.png".format(j),
                imgs.astype(np.uint8)
        )
