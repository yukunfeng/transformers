set -x
export GLUE_DIR=./glue_data/

# python download_glue_data.py --data_dir glue_data --tasks all

# task_name="sent_classify"
# python ./examples/text-classification/run_glue.py \
  # --model_name_or_path 'bert-base-chinese' \
  # --task_name $task_name \
  # --do_train \
  # --do_eval \
  # --do_predict \
  # --data_dir $GLUE_DIR/$task_name \
  # --max_seq_length 128 \
  # --per_device_train_batch_size 32 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 6.0 \
  # --save_steps 2000 \
  # --save_total_limit 1 \
  # --output_dir ./${task_name}_output/

# python score.py -gold_file glue_data/sent_classify/test.tsv -pred_file sent_classify_output/test_results_sent_classify.txt
# python ~/env_config/sending_emails.py -c "succ: $? zp classify"

task_name="sent_pair"
python ./examples/text-classification/run_glue.py \
  --model_name_or_path 'bert-base-chinese' \
  --task_name $task_name \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $GLUE_DIR/$task_name \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --save_steps 2000 \
  --save_total_limit 1 \
  --output_dir ./${task_name}_output/

python score.py -gold_file glue_data/sent_pair/test.tsv -pred_file sent_pair_output/test_results_sent_pair.txt
python ~/env_config/sending_emails.py -c "succ: $? zp pair"

# task_name=MRPC
# task_name=CoLA
# python ./examples/text-classification/run_glue.py \
  # --model_name_or_path 'roberta-base' \
  # --task_name $task_name \
  # --do_train \
  # --do_predict \
  # --do_eval \
  # --data_dir $GLUE_DIR/$task_name \
  # --max_seq_length 128 \
  # --per_device_train_batch_size 32 \
  # --learning_rate 2e-5 \
  # --num_train_epochs 3.0 \
  # --output_dir ./${task_name}_output/
