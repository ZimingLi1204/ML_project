import numpy as np
import imageio


def mask_visualize(
    img: np.ndarray,
    gt_mask: np.ndarray,
    gen_mask: np.ndarray,
    promt: np.ndarray,
    promt_type: str,
    img_number: int,
):
    if promt_type == "box":
        promt_img = np.moveaxis([img, img, img], 0, 2)
        if promt.shape == (4,):
            for x in range(promt[1], promt[3]):
                for y in range(promt[0], promt[2]):
                    promt_img[x][y][1] = 255
                    promt_img[x][y][0] = promt_img[x][y][2] = 0
        else:
            raise AssertionError
    elif promt_type == "single_point" or promt_type == "points":
        promt_img = np.moveaxis([img, img, img], 0, 2)
        if promt.ndim == 2 and promt.shape[1] == 2:
            for p in range(promt.shape[0]):
                promt_img[promt[p][0]][promt[p][1]][1] = 255
                promt_img[promt[p][0]][promt[p][1]][0] = 0
                promt_img[promt[p][0]][promt[p][1]][2] = 0
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
    img_output = np.concatenate((promt_img, mask_img, gt_img)).astype(np.uint8)
    imageio.imwrite(
        "../img/" + str(img_number) + ".png",
        img_output,
    )
