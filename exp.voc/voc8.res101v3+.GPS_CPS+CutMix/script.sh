# nvidia-smi

# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export volna="/home/GPS_semiseg/"
export NGPUS=8
export OUTPUT_PATH="/home/GPS_semiseg/log/aug2/"
export snapshot_dir=$OUTPUT_PATH

export batch_size=8
export learning_rate=0.00125
export snapshot_iter=1

# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_load_pre.py

export TARGET_DEVICE=$[$NGPUS-1]
python eval.py -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results 
# python eval.py -e 20-34 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results 

# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1