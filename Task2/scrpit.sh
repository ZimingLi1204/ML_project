python main.py --group_name finetune --save_name bc5_weighit_decay1e-3_adam \
--batch_size 5 --use_embedded --weight_decay 1.e-3 --optimizer Adam \
--gpu_id 2

python main.py --group_name finetune --save_name bc32_weighit_decay0 \
--batch_size 32 --use_embedded --weight_decay 0 \
--gpu_id 2