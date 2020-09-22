import os
import numpy as np
import pickle
import tqdm
from gensim.models import Word2Vec
import gensim.downloader as api


root = '/home/minz.s.won/data/pandora'
tags = np.load(os.path.join(root, 'tags.npy'))

tag_to_google_emb = {}
google_wv = api.load('word2vec-google-news-300')
for tag in tqdm.tqdm(tags):
	emb = google_wv.get_vector(tag)
	tag_to_google_emb[tag] = emb
pickle.dump(tag_to_google_emb, open(os.path.join(root, 'google_emb.pkl'), 'wb'))

tag_to_music_emb = {}
music_wv = Word2Vec.load(os.path.join(root, 'music_w2v', 'model_semeval_trigrams_300.model'))
for tag in tqdm.tqdm(tags):
	emb = music_wv.wv.get_vector(tag)
	tag_to_music_emb[tag] = emb
pickle.dump(tag_to_music_emb, open(os.path.join(root, 'music_emb.pkl'), 'wb'))
