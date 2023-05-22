python main.py --group_name finetune --save_name bc5_weighit_decay1e-3_adam_noembed \
--batch_size 5  --weight_decay 1.e-3 --optimizer Adam \
--gpu_id 0

python main.py --group_name finetune --save_name bc32_weighit_decay0 \
--batch_size 2 --weight_decay 0 \
--gpu_id 1

###test
python main.py --group_name debug --save_name test_dice \
--use_embedded \
--gpu_id 1 --test

python main.py --group_name debug --save_name debug \
--batch_size 2 --weight_decay 0 --use_embedded \
--gpu_id 1