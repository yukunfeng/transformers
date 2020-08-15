set -x
export GLUE_DIR=./glue_data/

task_name="tacred"
data_dir="$GLUE_DIR/$task_name"
output_dir="${task_name}_output/"
pred_file="$output_dir/test_results_${task_name}.txt"

batch_sizes=(32 16)
# lrs=(1e-5 2e-5 5e-6 1e-6)
lrs=(2e-5 5e-6 1e-6)
# warmups=(0 200 500 800 1000)
warmups=(200)
max_seq_lens=(324 256 128)
for batch_size in "${batch_sizes[@]}"
do
  for lr in "${lrs[@]}"
  do
    for warmup in "${warmups[@]}"
    do
        
        for max_seq_len in "${max_seq_lens[@]}"
        do
          rm -rf $output_dir
          python ./examples/text-classification/run_glue.py \
            --model_name_or_path 'roberta-base' \
            --task_name $task_name \
            --do_train \
            --do_eval \
            --do_predict \
            --data_dir $data_dir \
            --max_seq_length $max_seq_len \
            --per_device_train_batch_size $batch_size \
            --learning_rate $lr \
            --num_train_epochs 5 \
            --save_steps 2000 \
            --save_total_limit 1 \
            --logging_steps 1000 \
            --evaluate_during_training \
            --eval_steps 4000 \
            --warmup_steps $warmup \
            --seed 42 \
            --output_dir $output_dir

          echo "batch_size: $batch_size lr:$lr warmup:$warmup max_seq_len: $max_seq_len"
          python score.py -gold_file "$data_dir/test.json" -pred_file $pred_file
        done
        python ~/env_config/sending_emails.py -c "succ: $? tacred. max_seq_lens finished"
        exit 0
    done
  done
done


