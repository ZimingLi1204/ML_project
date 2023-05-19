from model import finetune_sam
import yaml
from config import process_cfg, parse_args
from utils import set_seed
from data import load_data_train, load_data_test
from torch.utils.data import DataLoader

if __name__ == '__main__':
    #load cfg
    cfg_file = open('config/cfg.yaml')
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)
    cfg_file.close()

    #load args
    args = parse_args()

    process_cfg(cfg, args, mode='train')

    # if cfg["use_wandb"]:
    #     wandb.init(
    #         project="tot-quant",
    #         group=cfg["train"]["group_name"],
    #         entity='zimingli1204',
    #         name = cfg["train"]["save_name"],
    #         config = cfg,
    #         sync_tensorboard=True,
    #         resume='allow',
    #     )
    #     wandb_writer = wandb
    # else:
    #     wandb_writer = None

    set_seed(cfg["random_seed"])

    algo = finetune_sam(cfg)

    print("#####Loading train data#######")
    train_dataset, val_dataset = load_data_train(cfg)
    print("#####Loading test data#######")
    test_dataset = load_data_test(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'])
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['test']['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['test']['batch_size'])
    print("#######Finishi Loading########")

    if cfg["test"]["test"]:
        algo.test(test_dataloader)
    else:
        algo.train(train_dataloader, val_dataloader)