python -u main.py \
--num_workers 16 \
--batch_size 256 \
--input_type 'cf' \
--w2v_type 'music' \
--is_balanced True \
--is_weighted True \
--data_path '/home/minz.s.won/data/pandora' \
--neptune_project 'minzwon/sandbox' \
--neptune_api_key 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWJkZGQzNTctNTllZC00ZDg0LTlkZTMtMWZkYThjZjRkMDQwIn0='
