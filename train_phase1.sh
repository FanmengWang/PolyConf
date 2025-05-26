save_dir="./phase1_ckpt" 
data_path="./dataset"
dict_name="dict.txt"
MASTER_PORT=8888
seed=0
n_gpu=8
lr=1e-4
epoch=600
warmup=0.06
dropout=0.1
batch_size=1
update_freq=1
pad_to_multiple=1
mar_diff_batch_mul=2

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
       --user-dir ./polyconf --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name --pad-to-multiple $pad_to_multiple --mar-diff-batch-mul $mar_diff_batch_mul \
       --task polyconf_phase1 --loss polyconf_phase1 --arch polyconf_phase1 \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --batch-size $batch_size --required-batch-size-multiple 1 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch  --pooler-dropout $dropout \
       --update-freq $update_freq --seed $seed \
       --log-interval 100 --log-format simple \
       --validate-interval 1 --patience 30 --keep-last-epochs 10 \
       --save-dir $save_dir --tmp-save-dir $save_dir/tmp --tensorboard-logdir $save_dir/tsb \
       --find-unused-parameters
