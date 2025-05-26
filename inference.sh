for gen_idx in $(seq 1 10)  
       do
              results_path="./infer_results/infer_results_${gen_idx}"
              weight_path="./phase2_ckpt/checkpoint_best.pt"
              data_path="./dataset"
              batch_size=1
              pad_to_multiple=1
              mar_num_ar_steps=64
              dict_name='dict.txt'

              export CUDA_VISIBLE_DEVICES=0  
              python ./polyconf/infer.py --user-dir ./polyconf $data_path --valid-subset test --mode 'infer' \
                     --results-path $results_path --seed $gen_idx \
                     --num-workers 8 --ddp-backend=c10d \
                     --batch-size $batch_size --pad-to-multiple $pad_to_multiple --required-batch-size-multiple 1 \
                     --task polyconf_inference --loss polyconf_inference --arch polyconf_inference \
                     --dict-name $dict_name --path $weight_path  \
                     --log-interval 10 --log-format simple --mar-num-ar-steps $mar_num_ar_steps
       done