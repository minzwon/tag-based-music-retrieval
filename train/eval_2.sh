nohup python -u main.py \
--mode 'TEST' \
--is_balanced True \
--is_weighted True \
--is_subset True \
--input_type 'hybrid' \
--w2v_type 'google' \
--data_path 'YOUR_DATA_PATH/sergio' \
--model_load_path './checkpoints/b-w-hybridp-google/CKPT_FILE_NAME' \
--neptune_project 'minzwon/pandora' \
--neptune_api_key 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWJkZGQzNTctNTllZC00ZDg0LTlkZTMtMWZkYThjZjRkMDQwIn0=' \
> 'logs/eval-b-w-hybridp-google.out' &
