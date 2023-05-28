python main.py --group_name finetune_single_point --save_name bc32_weighit_decay1e-3_adam_samloss_center_point \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type single_point \
--use_embedded --loss sam_loss \
--gpu_id 0

python main.py --group_name finetune_box --save_name bc32_weighit_decay1.e-2_adamW_samloss_20:1 \
--batch_size 32  --weight_decay 1.e-2 --optimizer AdamW --promt_type box \
--use_embedded --weight_list 20 1 --loss sam_loss \
--gpu_id 0

python main.py --group_name finetune_box --save_name bc32_weighit_decay1.e-3_adamW_samloss_20:1 \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type box \
--use_embedded --weight_list 20 1 --loss sam_loss --linear_warmup \
--gpu_id 0

python main.py --group_name finetune_points --save_name bc32_weighit_decay1e-3_adam_samloss_center_point \
--batch_size 32  --weight_decay 1.e-3 --optimizer AdamW --promt_type points \
--use_embedded --loss sam_loss \
--gpu_id 0


###test
python main.py --group_name debug --save_name test_promt \
--use_embedded --promt_type box \
--gpu_id 0 --test --data_root /BTCV_testset

python main.py --group_name debug --save_name test_dice \
--promt_type box \
--gpu_id 2 --test

python main.py --group_name debug --save_name debug \
--batch_size 2 --weight_decay 0 --use_embedded \
--gpu_id 1