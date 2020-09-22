import os
import tqdm
import pickle
import random
import glob
import numpy as np
import pandas as pd
import gensim.downloader as api
from collections import Counter

random.seed(0)

root = '/home/minz.s.won/data/pandora/'

filename = os.path.join(root, 'msd500', 'track_tags.tsv')
df = pd.read_csv(filename, sep='\t', names=['id', 'tag', 'merged', 'type', 'score'])
w2v = api.load('word2vec-google-news-300')

# get only exsisting item
existing_path = glob.glob(os.path.join(root, 'spec/*/*/*/*.npy'))
existing_ids = [line.split('/')[-1][:-4] for line in existing_path]
df = df[df.id.isin(existing_ids)]

# score threshold
df = df[df.score>=60]

# get top 50 tags
top100 = Counter(df.merged).most_common(200)
tags = [line[0] for line in top100]
available_tags = []
for tag in tags:
	try:
		v = w2v.get_vector(tag)
		available_tags.append(tag)
	except KeyError:
		continue
assert len(available_tags) > 100
tags = available_tags[:100]
for tag in tags:
	print(tag)
df = df[df.merged.isin(tags)]

# id to binary
tags = [tag.lower() for tag in tags]
tags.sort()
tag_to_index = {tags[i]: i for i in range(len(tags))}

ids = [msd_id for msd_id in set(df.id)]
ids.sort()
ids = np.array(ids)
np.save(open(os.path.join(root, 'existing_ids.npy'), 'wb'), ids)

binaries = np.zeros((len(ids), len(tags)))
i = 0
for msd_id in tqdm.tqdm(ids):
	annotations = list(df[df.id==msd_id].merged)
	for tag in annotations:
		binaries[i, tag_to_index[tag.lower()]] = 1
	i += 1


# iterative split
def split(threshold=100):
	is_run = True
	while is_run:
		ix = np.arange(len(ids))
		np.random.shuffle(ix)
		train_ix = ix[:int(len(ix)*0.7)]
		valid_ix = ix[int(len(ix)*0.7):int(len(ix)*0.85)]
		test_ix = ix[int(len(ix)*0.85):]
		train_count = binaries[train_ix].sum(axis=0)
		valid_count = binaries[valid_ix].sum(axis=0)
		test_count = binaries[test_ix].sum(axis=0)
		print(tags[train_count.argmin()], train_count.min())
		print(tags[valid_count.argmin()], valid_count.min())
		print(tags[test_count.argmin()], test_count.min())
		if np.min([train_count.min(), valid_count.min(), test_count.min()]) < threshold:
			is_run = True
		else:
			is_run = False
	return train_ix, valid_ix, test_ix

train_ix, valid_ix, test_ix = split(threshold=15)
train_ids = ['%s//%s'%(ix, ids[ix]) for ix in train_ix]
valid_ids = ['%s//%s'%(ix, ids[ix]) for ix in valid_ix]
test_ids = ['%s//%s'%(ix, ids[ix]) for ix in test_ix]

np.save(open(os.path.join(root, 'train_ids.npy'), 'wb'), train_ids)
np.save(open(os.path.join(root, 'valid_ids.npy'), 'wb'), valid_ids)
np.save(open(os.path.join(root, 'test_ids.npy'), 'wb'), test_ids)
np.save(open(os.path.join(root, 'binaries.npy'), 'wb'), binaries)
np.save(open(os.path.join(root, 'tags.npy'), 'wb'), tags)

# get tag-to-item dictionary (train set only)
train_tag_to_ix = {tag: [] for tag in tags}
for ix in train_ix:
	binary = binaries[ix]
	for ti, is_tag in enumerate(binary):
		if is_tag:
			train_tag_to_ix[tags[ti]].append('%s//%s'%(ix, ids[ix]))
pickle.dump(train_tag_to_ix, open(os.path.join(root, 'train_tag_to_ix.pkl'), 'wb'))

