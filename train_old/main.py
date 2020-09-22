import os
import pickle
import argparse
from solver import Solver
from data_loader import get_data_loader

def main(config):
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    train_loader = get_data_loader(config)

    solver = Solver(train_loader, config)
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=0.4)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--is_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_save_path', type=str, default='./model')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='/home/minz/ssd2/dataset/msd/')

    config = parser.parse_args()
    print(config)
    main(config)
