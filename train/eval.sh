python -u main.py \
--is_subset True \
--mode 'TEST' \
--input_type 'cf' \
--w2v_type 'google' \
--data_path 'YOUR_DATA_PATH' \
--model_load_path './checkpoints/b-w-cf-google/epoch=193.ckpt' \
--neptune_project 'YOUR_NEPTUNE_ID/YOUR_PROJECT_NAME' \
--neptune_api_key 'YOUR_NEPTUNE_API_KEY' \
