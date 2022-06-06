DATA_DIR=../../data/wikiNLDB
CHECKPOINT_DIR=../../checkpoints

python louvre_finetune.py \
    --train_save_step 1000 \
    --train_output_dir $CHECKPOINT_DIR/louvre_finetune_2 \
    --train_batch_size 150 \
    --eval_batch_size 64 \
    --num_train_epochs 50 \
    --train_file $DATA_DIR/wikinldb_mhop_train.json \
    --eval_file $DATA_DIR/wikinldb_mhop_dev.json 
