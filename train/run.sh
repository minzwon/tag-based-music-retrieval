nohup python -u main.py \
--input_type 'spec' \
--w2v_type 'google' \
--data_path '/home/minz.s.won/data/pandora' \
--model_save_path './checkpoints/nb-nw-spec-google' \
--neptune_project 'minzwon/pandora' \
--neptune_api_key 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWJkZGQzNTctNTllZC00ZDg0LTlkZTMtMWZkYThjZjRkMDQwIn0=' \
> 'logs/nb-nw-spec-google.out' &
