import os
import torch
import argparse
from solver import Solver
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger

def main(config):
	solver = Solver(config)
	logger = NeptuneLogger(project_name=config.neptune_project, api_key=config.neptune_api_key)
	checkpoint_callback = ModelCheckpoint(filepath=config.model_save_path,
										  save_top_k=1,
										  verbose=True,
										  monitor="map",
										  mode="max",
										  prefix="")
	trainer = Trainer(default_root_dir=config.model_save_path,
					  gpus=config.gpu_id,
					  logger=logger,
					  checkpoint_callback=checkpoint_callback,
					  max_epochs=config.n_epochs)
	if config.mode == 'TRAIN':
		trainer.fit(solver)
		trainer.save_checkpoint(os.path.join(config.model_save_path, 'last.ckpt'))
	elif config.mode == 'TEST':
		S = torch.load(config.model_load_path)['state_dict']
		SS = {key[6:]: S[key] for key in S.keys()}
		solver.model.load_state_dict(SS)
		trainer.test(solver)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--mode', type=str, default='TRAIN', choices=['TRAIN', 'TEST'])

	# model parameters
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_chunk', type=int, default=8)
	parser.add_argument('--input_length', type=int, default=173)
	parser.add_argument('--margin', type=float, default=0.4)
	parser.add_argument('--input_type', type=str, default='spec', choices=['spec', 'cf', 'hybrid'])
	parser.add_argument('--w2v_type', type=str, default='google', choices=['google', 'music'])
	parser.add_argument('--is_balanced', type=bool, default=False)
	parser.add_argument('--is_weighted', type=bool, default=False)
	parser.add_argument('--is_subset', type=bool, default=False)

	# training parameters
	parser.add_argument('--n_epochs', type=int, default=200)
	parser.add_argument('--gpu_id', type=str, default='0')
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--model_save_path', type=str, default='./checkpoints')
	parser.add_argument('--model_load_path', type=str, default='.')
	parser.add_argument('--data_path', type=str, default='.')
	parser.add_argument('--neptune_project', type=str, default='.')
	parser.add_argument('--neptune_api_key', type=str, default='.')

	config = parser.parse_args()

	print(config)
	main(config)
