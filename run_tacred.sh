set -x
export GLUE_DIR=/home/lr/yukun/kg-bert/ERNIE/data/

task_name="fewrel"
data_dir="$GLUE_DIR/tacred"
output_dir="tacred_output/"

batch_sizes=(32 16)
lrs=(1e-5 2e-5 5e-6 1e-6)
warmups=(0 200 500 800 1000)
for batch_size in "${batch_sizes[@]}"
do
  for lr in "${lrs[@]}"
  do
    for warmup in "${warmups[@]}"
    do

      rm -rf $output_dir
      python ./examples/text-classification/run_glue.py \
        --model_name_or_path 'roberta-base' \
        --task_name $task_name \
        --do_train \
        --do_eval \
        --do_predict \
        --data_dir $data_dir \
        --max_seq_length 184 \
        --per_device_train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs 5 \
        --save_steps 2000 \
        --save_total_limit 1 \
        --logging_steps 1000 \
        --evaluate_during_training \
        --eval_steps 1000 \
        --warmup_steps $warmup \
        --seed 42 \
        --output_dir $output_dir

      echo "batch_size: $batch_size lr:$lr warmup:$warmup"
      python score.py -gold_file "$data_dir/test.json" -pred_file "$output_dir/test_results_${task_name}.txt"
      exit 0 
      python ~/env_config/sending_emails.py -c "succ: $? tacred. Warmup finished"
    done
    exit 0
    python ~/env_config/sending_emails.py -c "succ: $? tacred. Warmup finished"
  done
done


