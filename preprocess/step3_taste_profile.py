import os
import tqdm
import fire
import numpy as np
import pandas as pd


class Processor:
    def __init__(self):
        print('get collaborative filtering embeddings..')

    def run(self, root='YOUR_DATA_PATH'):
        user_item_emb = np.load(os.path.join(root, 'item_factors_MSD.npy'))
        user_item_ix = pd.read_csv(os.path.join(root, 'items_index_MSD.tsv'), sep='\t', names=['msd_id'])
        sorted_ix = np.load(os.path.join(root, 'existing_ids.npy'))

        sorted_emb = []
        for msd_id in tqdm.tqdm(sorted_ix):
            ix = user_item_ix.loc[user_item_ix.msd_id==msd_id].index[0]
            sorted_emb.append(user_item_emb[ix])
        np.save(open(os.path.join(root, 'ix_to_cf.npy'), 'wb'), sorted_emb)


if __name__ == '__main__':
    p = Processor()
    fire.Fire({'run': p.run})
