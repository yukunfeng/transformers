set -x
export GLUE_DIR=./glue_data/

# python download_glue_data.py --data_dir glue_data --tasks all

task_name=CoLA
# task_name=SST-2
python ./examples/text-classification/run_glue.py \
  --model_name_or_path 'roberta-base' \
  --task_name $task_name \
  --do_train \
  --do_predict \
  --do_eval \
  --data_dir $GLUE_DIR/$task_name \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./${task_name}_output/
