python main.py --group_name finetune_box --save_name bc32_wd1.e-4_adamW_samloss_2:1_lrdecay_warmup_iouloss0.05_Step1800_multimask_max \
--batch_size 32  --weight_decay 1.e-4 --optimizer AdamW --promt_type box \
--use_embedded --weight_list 2 1 --loss sam_loss --lr_schedular StepLR \
--step_size 1800 --multimask max --linear_warmup \
--iou_scale 0.05 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_box --save_name bc32_wd1.e-2_adamW_samloss_2:1_lrdecay_iouloss0.05_Step1800_multimask_max \
--batch_size 32  --weight_decay 1.e-2 --optimizer AdamW --promt_type box \
--use_embedded --weight_list 2 1 --loss sam_loss --lr_schedular StepLR \
--step_size 1800 --multimask max \
--iou_scale 0.05 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_box --save_name bc32_wd1.e-3_adamW_samloss_1:1_lrdecay_iouloss0.05_Step1800 \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type box \
--use_embedded --weight_list 1 1 --loss sam_loss --lr_schedular StepLR --step_size 1800 \
--iou_scale 0.05 --log_dir /log \
--gpu_id 0

python main.py --group_name finetune_single_point --save_name bc32_wd1e-3_adamW_samloss_2:1_lrdecay_StepLR2000_iouloss0.05_multimask_max \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type single_point \
--use_embedded --weight_list 2 1 --loss sam_loss --linear_warmup --lr_schedular StepLR \
--step_size 2000 --multimask max \
--iou_scale 0.05 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_points --save_name bc32_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0_8_8_multimask_min \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type points \
--use_embedded --weight_list 1 1 --loss sam_loss --lr_schedular StepLR \
--step_size 2000 --multimask min \
--iou_scale 0.05 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_points --save_name bc32_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0_8_8_center_multimask_max \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type points --center_point \
--use_embedded --weight_list 1 1 --loss sam_loss --lr_schedular StepLR \
--step_size 2000 --multimask max \
--iou_scale 0.05 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_grid_points --save_name bc16_wd1e-3_adamW_samloss_1:1_lrdecay_StepLR2000_iouloss0.05_12points_multimask_max \
--batch_size 16  --weight_decay 1.e-3 --optimizer AdamW --promt_type grid_points \
--use_embedded --weight_list 1 1 --loss sam_loss --lr_schedular StepLR \
--step_size 2000 --multimask max \
--iou_scale 0.05 --point_num 12 --log_dir log --model_root /pretrain_model \
--gpu_id 0

python main.py --group_name finetune_grid_points --save_name bc16_wd1e-3_adamW_samloss_1:1_lrdecay_StepLR2000_warmup_iouloss0.05_24points_multimask_max \
--batch_size 16  --weight_decay 1.e-3 --optimizer AdamW --promt_type grid_points \
--use_embedded --weight_list 1 1 --loss sam_loss --lr_schedular StepLR \
--step_size 2000 --linear_warmup --multimask max \
--iou_scale 0.05 --point_num 24 --log_dir log --model_root /pretrain_model \
--gpu_id 0 

###test
python main.py --group_name test --save_name test \
--use_embedded --promt_type box \
--ckpt_path /root/autodl-tmp/ML_project/decoder_checkpoint/box_8643.pth \
--gpu_id 0 --test --data_root /BTCV_testset --model_root /pretrain_model 

python main.py --group_name test --save_name test \
--use_embedded --promt_type points --center_point \
--ckpt_path /root/autodl-tmp/ML_project/decoder_checkpoint/points_center_8110.pth \
--gpu_id 0 --test --data_root /BTCV_testset --model_root /pretrain_model 

python main.py --group_name test --save_name test \
--use_embedded --promt_type grid_points --point_num 12 \
--ckpt_path /root/autodl-tmp/ML_project/decoder_checkpoint/grid_points_12_7788.pth \
--gpu_id 0 --test --data_root /BTCV_testset --model_root /pretrain_model 

python main.py --group_name test --save_name test \
--use_embedded --promt_type single_point \
--ckpt_path /root/autodl-tmp/ML_project/decoder_checkpoint/single_point_6817.pth \
--gpu_id 0 --test --data_root /BTCV_testset --model_root /pretrain_model 


box 86.05 /log/finetune_box/bc32_wd1.e-3_adamW_samloss_2:1_lrdecay_iouloss0.05/12.pth (step_size 1800)

single_point_center 68.41 log/finetune_single_point/bc32_wd1e-3_adamW_samloss_10:1_lrdecay_iouloss0.05_center/11.pth 
single_point 63.08 log/finetune_single_point/bc32_wd1e-3_adamW_samloss_10:1_lrdecay_iouloss0.05/11.pth

points 81.70 log/finetune_points/bc32_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0_8_8/19.pth
points_center 78.6 log/finetune_points/bc32_wd1e-3_adamW_center_samloss_1:1_lrdecay_iouloss0_8_8/12.pth

grid_points_16 80.18 log/finetune_grid_points/bc16_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0.05_16points/14.pth
grid_points_24 83.45 log/finetune_grid_points/bc16_wd1e-3_adamW_samloss_1:1_lrdecay_warmup_iouloss0.05_24points/10.pth
grid_points_12 73.26 log/finetune_grid_points/bc16_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0.05_12points/12.pth

multimask_training:
box 86.43 log/finetune_box/bc32_wd1.e-3_adamW_samloss_2:1_lrdecay_iouloss0.05_Step1800_multimask_max/14.pth

single_point_center 73.48 log/finetune_single_point/bc32_wd1e-3_adamW_samloss_5:1_lrdecay_iouloss0.05_StepLR2000_center_multimask_max/18.pth

single_point 68.17 log/finetune_single_point/bc32_wd1e-3_adamW_samloss_5:1_lrdecay_iouloss0.05_StepLR2000_center_multimask_max/17.pth

points 84.38 log/finetune_points/bc32_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0_8_8_multimask_max/18.pth
points_center 81.1 log/finetune_points/bc32_wd1e-3_adamW_samloss_1:1_lrdecay_iouloss0_8_8_center_multimask_max/15.pth

grid_points_16 80.72 log/finetune_grid_points/bc16_wd1e-4_adamW_samloss_1:1_lrdecay_iouloss0.05_16points_multimask_max/10.pth
grid_points_24 83.83 log/finetune_grid_points/bc16_wd1e-3_adamW_samloss_1:1_lrdecay_StepLR2000_warmup_iouloss0.05_24points_multimask_max/11.pth
grid_points_12 77.88 log/finetune_grid_points/bc16_wd1e-3_adamW_samloss_1:1_lrdecay_StepLR2000_iouloss0.05_12points_multimask_max/9.pth