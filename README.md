# ML_project
PKU machine learning project 2023 spring

# Installation
**follow installation guidance in [segment-anything](https://github.com/facebookresearch/segment-anything) project.**
* python 3.7
* pytorch 1.7.1
* CUDA capable GPU (one is enough)

# Preparation
**download preprocessed CT data and fine-tuned model weights from this [link](https://disk.pku.edu.cn:443/link/5265EA11237F0D76F7C8FEF487DCDA24)**

* place model weights in `./pre_train/`

* place data for train and test in `./BTCV/`

# Task1
**Zero-shot performance on test set**

`cd Task2`

`python main.py --promt_type box --test`

**You may also define:**
* `--device_id` id.
* `--seed` random seed.
* `--center_point` choose center point prompt or random choose point prompt
* `--ckpt_path` path/to/your/sam_model_decoder_weights

**you can also modify these options in Task2/config/cfg.yaml**

# Task2
**Fine-tune on training set**


# Task3

