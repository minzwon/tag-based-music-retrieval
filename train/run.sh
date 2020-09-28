nohup python -u main.py \
--is_weighted True \
--is_balanced True \
--is_subset True \
--input_type 'spec' \
--w2v_type 'google' \
--data_path '/home/minz.s.won/data/pandora' \
--model_save_path './checkpoints/b-w-specp-google' \
--neptune_project 'minzwon/pandora' \
--neptune_api_key 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWJkZGQzNTctNTllZC00ZDg0LTlkZTMtMWZkYThjZjRkMDQwIn0=' \
> 'logs/b-w-specp-google.out' &
