set -x
export GLUE_DIR=/home/lr/yukun/kg-bert/ERNIE/data/

task_name="tacred"
data_dir="$GLUE_DIR/$task_name"
output_dir="${task_name}_output/"

  # --model_name_or_path 'roberta-base' \
python ./examples/text-classification/run_glue.py \
  --model_name_or_path 'bert-base-uncased' \
  --task_name 'fewrel' \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $data_dir \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --save_steps 2000 \
  --save_total_limit 1 \
  --output_dir $output_dir

python score.py -gold_file "$data_dir/test.json" -pred_file "$output_dir/test_results_${task_name}.txt"
python ~/env_config/sending_emails.py -c "succ: $? tacred"
