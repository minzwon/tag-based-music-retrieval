nohup python -u main.py \
--is_balanced True \
--is_weighted True \
--input_type 'cf' \
--w2v_type 'google' \
--data_path '/home/minz.s.won/data/' \
--model_save_path './checkpoints/b-w-cf-google' \
--neptune_project 'YOUR_NEPTUNE_ID/YOUR_PROJECT_NAME' \
--neptune_api_key 'YOUR_NEPTUNE_API_KEY' \
> 'logs/b-w-cf-google.out' &
