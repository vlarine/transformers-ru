export SQUAD_DIR=~/data/sber_SQuAD/

python3 run_squad.py \
  --model_type bert \
  --config_name ./rubert_cased_L-12_H-768_A-12_v1/bert_config.json \
  --tokenizer_name ./rubert_cased_L-12_H-768_A-12_v1/vocab.txt \
  --model_name_or_path ./rubert_cased_L-12_H-768_A-12_v1/bert_model.bin \
  --do_train \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --save_steps 2000 \
  --overwrite_cache \
  --overwrite_output_dir \
  --output_dir /tmp/debug_squad/
