DATA_DIR=../../data/hotpotQA
CHECKPOINT_DIR=../../checkpoints
WIKICORPUS_DIR=../../data/wiki_corpus
OUTPUT_DIR=../../outputs

TOKENIZERS_PARALLELISM=false python nldb_pred.py \
    --pred_batch_size 512 \
    --init_checkpoint $CHECKPOINT_DIR/louvre_finetune_2_best/ \
    --topk 1 \
    --beam_size 2 \
    --input_file /data/private/NeuralDB/WikiNLDB/v2.4_25/test.jsonl \
    --pred_save_file 25.json 
