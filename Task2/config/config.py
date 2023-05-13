import yaml
import os
import numpy as np
import datetime
import torch
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='train args')

    # #DATA
    # parser.add_argument('--train_feature_length', type=int, default=None, help='how many days used before for features')
    # parser.add_argument('--train_length', type=int, default=None, help='how many trading days used before for features')
    # parser.add_argument('--use_dataloder', default=False, action='store_true', help='use dataloder or batch data ')
    
    # #learning
    # parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    # parser.add_argument('--batch_split', type=int, default=None, help='batch split for no dataloder situation')
    # parser.add_argument('--learning_rate', type=float, default=None)
    # parser.add_argument('--weight_decay', default=None, type=float)
    # parser.add_argument('--schedular', default=None, type=str)

    # #loss 
    # parser.add_argument('--cls_loss_weight', type=float, default=None, help='loss weight of classifier')
    # parser.add_argument('--domain_loss_weight', type=float, default=None, help='loss weight of DAN (if use)')
    # parser.add_argument('--loss', type=str, default=None)
    # parser.add_argument('--beta', type=float, default=None, help= "beta of smoothl1 loss")
    
    # #eval
    # parser.add_argument('--eval_freq', type=int, default=None)
    # parser.add_argument('--eval_time', type=int, default=None)

    # #model
    # parser.add_argument('--linear', default=False, action='store_true')
    # parser.add_argument('--debug', default=False, action='store_true')
    # parser.add_argument('--use_dropout', default=False, action='store_true')
    # parser.add_argument('--use_bn', default=False, action='store_true')

    # #os path
    # parser.add_argument('--data_root', default=None, type=str, help='train data root')
    # parser.add_argument('--ckpt_path', default=None, type=str, help='load model dir')
    # parser.add_argument('--save_name', type=str, default=None)
    # parser.add_argument('--group_name', type=str, default=None)

    # #others
    # parser.add_argument('--seed', type=int, default=None)
    # parser.add_argument('--use_wandb', default=False, action='store_true')
    # parser.add_argument('--test', default=False, action='store_true')

    # #domain adversarial
    # parser.add_argument('--use_DA', default=False, action='store_true')
    # parser.add_argument('--domain_class', type=int, default=None)

    args = parser.parse_args()
    return args

def process_cfg(cfg, args=None, mode = ''):

    #处理args
    ''' if args is not None:

    
        #data
        if args.train_length is not None:
            cfg["train"]["train_length"] = args.train_length
        if args.train_feature_length is not None:
            cfg["train"]["train_feature_length"] = args.train_feature_length
        if args.use_dataloder is not None:
            cfg["train"]["use_dataloder"] = args.use_dataloder

        
        #改了其中一个， 需要重新生成gt data
        if (args.train_length is not None) or (args.train_feature_length is not None):
            cfg["data"]["use_saved_gt"] = False

        #learning
        if args.batch_size is not None:
            cfg["train"]["batch_size"] = args.batch_size     
        if args.batch_split is not None:
            cfg["train"]["batch_split"] = args.batch_split
        if args.learning_rate is not None:
            cfg["train"]["learning_rate"] = args.learning_rate    
        if args.weight_decay is not None:
            cfg["train"]["weight_decay"] = args.weight_decay
        if args.schedular is not None:
            cfg["train"]["schedular"] = args.schedular

        #loss    
        if args.cls_loss_weight is not None:
            cfg["train"]["cls_loss_weight"] = args.cls_loss_weight
        if args.loss is not None:
            cfg["train"]["loss"] = args.loss
        if args.beta is not None:
            cfg["train"]["beta"] = args.beta

        #eval    
        if args.eval_freq is not None:
            cfg["train"]["eval_freq"] = args.eval_freq
            cfg["train"]["save_freq"] = cfg["train"]["eval_freq"] 
        if args.eval_time is not None:
            cfg["train"]["eval_time"] = args.eval_time

        #model
        if args.debug:
            cfg["debug"] = True
            cfg["train"]["epochs"] = 10000
            cfg["train"]["use_dataloder"] = True
        if args.linear:
            cfg["model"]["Linear_model"] = True
        if args.use_dropout:
            cfg["model"]["use_dropout"] = True
        if args.use_bn:
            cfg["model"]["use_bn"] = True

        #os path
        if args.save_name is not None:
            cfg["train"]["save_name"] = args.save_name
        if args.group_name is not None:
            cfg["train"]["group_name"] = args.group_name
        if args.data_root is not None:
            cfg["data"]["load_data_root"] = args.data_root
        if args.ckpt_path is not None:
            cfg["train"]["ckpt_path"] = args.ckpt_path

        #others
        if args.seed is not None:
            cfg["random_seed"] = args.seed
        if args.use_wandb:
            cfg["use_wandb"] = True
        if args.test:
            cfg["test"]["test"] = True

        #domain adversarial
        if args.use_DA:
            cfg["model"]["use_domain_adversarial"] = True
        if args.domain_class is not None:
            cfg["model"]["domain_head"]["domain_class"] = args.domain_class

        if args.ckpt_path is not None:
            d = r'\\'
            cfg["train"]["save_name"] = args.ckpt_path.split(d[0])[-2]
        else:
            cfg["train"]["save_name"] = cfg["train"]["save_name"] + '_' + str(datetime.datetime.now().strftime("%H-%M-%S"))
        cfg["train"]["log_root"] = os.path.join(cfg["train"]["log_root"], str(datetime.date.today()), cfg["train"]["save_name"])

    if torch.cuda.is_available():
        cfg["device"] = "cuda:0"
    else:
        cfg["device"] = "cpu"

    #更改保存log的路径
    # pdb.set_trace()
    

    flag = 0
    for p in ['train_gt_rate', 'val_gt_rate', 'train_index', 'val_index']:
        if os.path.exists(os.path.join(cfg["data"]["save_data_root"], p + '.npy')):
            continue
        else:
            flag = 1
    if (flag == 1):
        cfg["data"]["use_saved_gt"] = False
    

    cfg["model"]["daily_head"]["train_length"] = cfg["train"]["train_length"]
    cfg["model"]["feature_head"]["train_length"] = cfg["train"]["train_feature_length"]

    if mode == 'statistic' or mode == 'visualization':
        cfg["data"]["use_saved_gt"] = True
        cfg["data"]["use_numpy_data"] = True
        cfg['data']["pre_process"] = False
    elif mode == 'train':
        cfg["data"]["use_saved_gt"] = True
        cfg["data"]["use_numpy_data"] = True
        cfg['data']["pre_process"] = False
        # cfg['data']['load_data_root'] = r'C:\Users\pc\Desktop\ziming\data\npy_data'
    elif mode == 'process_data':
        cfg["data"]["use_saved_gt"] = False
        cfg["data"]["use_numpy_data"] = True
        cfg['data']["pre_process"] = True

    #根据setting设置data root
    if cfg['data']['pre_process'] and not cfg["test"]["test"]:
        cfg["data"]["load_data_root"] = r'C:\Users\pc\Desktop\ziming\data\raw_npy_data'
    # if cfg["train"]["use_dataloder"] == False:
    #     if mode == 'process_data':
    #         cfg["data"]["save_data_root"] = r'C:\Users\pc\Desktop\ziming\data\date_batch_npy_data'
    #     else:
    #         cfg["data"]["load_data_root"] = r'C:\Users\pc\Desktop\ziming\data\date_batch_npy_data'

    assert(cfg["train"]["loss"] in ["MSE", "SmoothL1"])
    '''
        
    if not os.path.exist(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])

    print(cfg)
    return cfg