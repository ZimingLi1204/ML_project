python main.py --group_name finetune_box --save_name bc4_weighit_decay1e-3_adam_noembed \
--batch_size 4  --weight_decay 1.e-3 --optimizer Adam --promt_type box \
--gpu_id 0

python main.py --group_name finetune --save_name bc32_weighit_decay0 \
--batch_size 2 --weight_decay 0 \
--gpu_id 1

###test
python main.py --group_name debug --save_name test_dice \
--use_embedded --promt_type box \
--gpu_id 1 --test

python main.py --group_name debug --save_name test_dice \
--promt_type box \
--gpu_id 2 --test

python main.py --group_name debug --save_name debug \
--batch_size 2 --weight_decay 0 --use_embedded \
--gpu_id 1