import yaml
import os
import numpy as np
import datetime
import torch
import argparse
import pdb

def parse_args():

    parser = argparse.ArgumentParser(description='train args')

    # #DATA
    # parser.add_argument('--train_feature_length', type=int, default=None, help='how many days used before for features')
    # parser.add_argument('--train_length', type=int, default=None, help='how many trading days used before for features')
    # parser.add_argument('--use_dataloder', default=False, action='store_true', help='use dataloder or batch data ')
    parser.add_argument('--data_root', default=None, type=str, help='train data root')
    parser.add_argument('--use_embedded', default=False, action='store_true')
    parser.add_argument('--load_from_disk', default=False, action='store_true')

    #promt
    parser.add_argument('--promt_type', default=None, type=str)
    parser.add_argument('--center_point', default=False, action='store_true')
    parser.add_argument('--point_num', default=None, type=int)
    parser.add_argument('--point_size', default=None, type=int)

    # #learning
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    # parser.add_argument('--batch_split', type=int, default=None, help='batch split for no dataloder situation')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--optimizer', default=None, type=str)

    parser.add_argument('--linear_warmup', default=False, action='store_true')
    parser.add_argument('--warmup_iter', type=float, default=None)
    parser.add_argument('--start_factor', type=float, default=None)
    
    parser.add_argument('--lr_schedular', default=None, type=str)
    parser.add_argument('--schedular_gamma', default=None, type=float)
    parser.add_argument('--step_size', default=None, type=int)

    # #loss 
    # parser.add_argument('--cls_loss_weight', type=float, default=None, help='loss weight of classifier')
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--weight_list', nargs='+', default=None, type=int)
    parser.add_argument('--iou_scale', default=None, type=float)
    parser.add_argument('--multimask', type=str, default=None)
    
    # #eval
    # parser.add_argument('--eval_freq', type=int, default=None)
    # parser.add_argument('--eval_time', type=int, default=None)

    # #model
    # parser.add_argument('--linear', default=False, action='store_true')
    # parser.add_argument('--debug', default=False, action='store_true')
    # parser.add_argument('--use_dropout', default=False, action='store_true')
    # parser.add_argument('--use_bn', default=False, action='store_true')
    parser.add_argument('--model_root', default=None, type=str, help='pretrain vit model dir')

    # #os path
    parser.add_argument('--ckpt_path', default=None, type=str, help='load model dir')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # #others
    parser.add_argument('--seed', type=int, default=None)
    # parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu_id', default=None, type=int)

    # #domain adversarial
    # parser.add_argument('--use_DA', default=False, action='store_true')
    # parser.add_argument('--domain_class', type=int, default=None)

    args = parser.parse_args()
    return args

def process_cfg(cfg, args=None, mode = ''):

    #处理args

    assert(cfg["train"]["loss"] in ["MSE", "SmoothL1"])

    #train hyperparameters
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['train']['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        cfg['train']['weight_decay'] = args.weight_decay
    if args.optimizer is not None:
        cfg['train']['optimizer'] = args.optimizer

    if args.linear_warmup:
        cfg["train"]["linear_warmup"] = True
    if args.start_factor:
        cfg["train"]["start_factor"] = args.start_factor
    if args.warmup_iter:
        cfg["train"]["warmup_iter"] = args.warmup_iter

    if args.lr_schedular is not None:
        cfg['train']["lr_schedular"] = args.lr_schedular
        cfg["train"]["lr_decay"] = True
    if args.schedular_gamma is not None:
        cfg['train']["schedular_gamma"] = args.schedular_gamma
    if args.step_size is not None:
        cfg['train']["step_size"] = args.step_size

    #####loss
    if args.loss is not None:
        cfg["train"]["loss"] = args.loss
    if args.weight_list is not None:
        cfg["train"]["weight_list"] = args.weight_list
    if args.iou_scale is not None:
        cfg["train"]["iou_scale"] = args.iou_scale

    if args.multimask is not None:
        cfg["train"]["multimask"] = args.multimask

        
    ####data
    if args.data_root is not None:
        cfg['data']['data_root'] = args.data_root
    if args.use_embedded:
        cfg["data"]["use_embedded"] = True
    if args.load_from_disk:
        cfg["data"]["load_from_disk"] = True

    ####promt
    if args.promt_type is not None:
        cfg["promt"]["promt_type"] = args.promt_type
    if args.center_point:
        cfg["promt"]["center_point"] = True
    if args.point_num is not None:
        cfg["promt"]["point_num"] = args.point_num
    if args.point_size is not None:
        cfg["promt"]["point_size"] = args.point_size

    ###model
    if args.model_root is not None:
        cfg['model']['model_root'] = args.model_root

    ###others
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    if args.gpu_id is not None:
        cfg["device_id"] = args.gpu_id
    if args.test:
        cfg["test"]["test"] = True
    
    ###save name and dir
    if args.ckpt_path is not None:
        cfg['model']['decoder_path'] = args.ckpt_path
        cfg["model"]["load_decoder"] = True
    if args.log_dir is not None:
        cfg['train']['log_dir'] = args.log_dir
    if args.group_name is not None:
        cfg['train']['group'] = args.group_name
    if args.save_name is not None:
        cfg['train']['save_name'] = args.save_name


    ##create log path to save log
    # pdb.set_trace()
    if not os.path.exists(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])

    cfg['train']['log_dir'] = os.path.join(cfg['train']['log_dir'], cfg["train"]['group'])
    if not os.path.exists(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])

    cfg['train']['log_dir'] = os.path.join(cfg['train']['log_dir'], cfg["train"]['save_name'])
    if not os.path.exists(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])


    ####data
    if cfg["data"]["use_embedded"]:
        cfg["data"]["data_name"] = "vit-h_embedding_bc1"


    print(cfg)
    return cfg