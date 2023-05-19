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
    parser.add_argument('--use_embedded', default=False, action='store_true')
    
    # #learning
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    # parser.add_argument('--batch_split', type=int, default=None, help='batch split for no dataloder situation')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--optimizer', default=None, type=str)
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
    parser.add_argument('--data_root', default=None, type=str, help='train data root')
    parser.add_argument('--ckpt_path', default=None, type=str, help='load model dir')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--gpu_name', type=str, default=None)

    # #others
    # parser.add_argument('--seed', type=int, default=None)
    # parser.add_argument('--use_wandb', default=False, action='store_true')
    # parser.add_argument('--test', default=False, action='store_true')
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

    ####data

    cfg["data"]["use_embedded"] = True

    ###others
    if args.gpu_id is not None:
        cfg["device_id"] = args.gpu_id
    
    ###save name and dir
    if args.group_name is not None:
        cfg['train']['group'] = args.group_name
    if args.save_name is not None:
        cfg['train']['save_name'] = args.save_name

    cfg['train']['log_dir'] = os.path.join(cfg['train']['log_dir'], cfg["train"]['group'])
    if not os.path.exists(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])

    cfg['train']['log_dir'] = os.path.join(cfg['train']['log_dir'], cfg["train"]['save_name'])
    if not os.path.exists(cfg['train']['log_dir']):
        os.mkdir(cfg['train']['log_dir'])


    ####data
    if cfg["data"]["use_embedded"]:
        cfg["data"]["data_name"] = "vit-h_embedding_dataset1"


    print(cfg)
    return cfg