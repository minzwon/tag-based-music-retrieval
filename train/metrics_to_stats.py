import numpy as np
import pickle
import pandas as pd
from collections import Counter

tt = pd.read_csv('../msd_sub/track_tags.tsv', sep='\t', names=['id', 'tag', 'cleaned', 'type', 'number'])
metrics = pickle.load(open('metrics.pkl', 'rb'))
tags = list(set(tt.cleaned))
tags.sort()

tag_to_cat = {}
cats = []
for _tag in tags:
    _cat = list(tt[tt.cleaned==_tag].type)[0]
    tag_to_cat[_tag] = _cat
    cats.append(_cat)

cnt = Counter(cats)
print(cnt)

cats = list(set(cats))
cats.sort()

cat_to_metric = {}
for _cat in cats:
    cat_to_metric[_cat] = []

for i in range(len(tags)):
    _cat = tag_to_cat[tags[i]]
    _score = metrics['pr_auc'][i]
    cat_to_metric[_cat].append(_score)

for _cat in cats:
    print('%s: %.4f' % (_cat, np.mean(cat_to_metric[_cat])))


print('merge location and language/origin')

cat_to_metric = {}
for _cat in cats:
    cat_to_metric[_cat] = []

for i in range(len(tags)):
    _cat = tag_to_cat[tags[i]]
    if _cat == 'location':
        _cat = 'language/origin'
    _score = metrics['pr_auc'][i]
    cat_to_metric[_cat].append(_score)

for _cat in cats:
    if _cat != 'location':
        print('%s: %.4f' % (_cat, np.mean(cat_to_metric[_cat])))


