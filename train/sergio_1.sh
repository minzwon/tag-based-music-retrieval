mkdir checkpoints/b-w-sub-cfp-google
nohup python -u main.py \
--is_balanced True \
--is_weighted True \
--is_subset True \
--input_type 'cf' \
--w2v_type 'google' \
--data_path 'YOUR_DATA_PATH/sergio' \
--model_save_path './checkpoints/b-w-sub-cfp-google' \
--neptune_project 'minzwon/pandora' \
--neptune_api_key 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWJkZGQzNTctNTllZC00ZDg0LTlkZTMtMWZkYThjZjRkMDQwIn0=' \
> 'logs/b-w-sub-cfp-google.out' &
