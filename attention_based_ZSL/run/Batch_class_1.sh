data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds
exp_dir=outputs/Batch_class_1
cfg_file=Confg/Batch_class_1.yml
seed=200
lr_S=0.001
lr_I=0.001
wd=1e-3

python3 run.py --data_path $data_path \
              --exp_dir $exp_dir \
			  --cfg_file $cfg_file\
			  --lr_S $lr_S\
			  --lr_I $lr_I\
			  --manualSeed $seed\
			  --weight-decay $wd