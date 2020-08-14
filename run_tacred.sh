set -x
export GLUE_DIR=/home/lr/yukun/kg-bert/ERNIE/data/

task_name="fewrel"
data_dir="$GLUE_DIR/tacred"
output_dir="tacred_output/"
rm -rf $output_dir

python ./examples/text-classification/run_glue.py \
  --model_name_or_path 'roberta-base' \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $data_dir \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --save_steps 2000 \
  --save_total_limit 1 \
  --logging_steps 1000 \
  --evaluate_during_training \
  --eval_steps 1000 \
  --warmup_steps 500 \
  --output_dir $output_dir

python score.py -gold_file "$data_dir/test.json" -pred_file "$output_dir/test_results_${task_name}.txt"
python ~/env_config/sending_emails.py -c "succ: $? tacred"
