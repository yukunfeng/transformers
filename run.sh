set -x
export GLUE_DIR=./glue_data/

# python download_glue_data.py --data_dir glue_data --tasks all

task_name="sent_classify"
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
  --num_train_epochs 3.0 \
  --output_dir ./${task_name}_output/

python ~/env_config/sending_emails.py -c '2020-07-25 21:15 zp classify'
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
