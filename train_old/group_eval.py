import numpy as np
import pandas as pd
import pickle

# load files
metrics = pickle.load(open('metrics.pkl', 'rb'))
selected_tags = pd.read_csv('../msd_sub/selected_tags.tsv', sep='\t', names=['tag', 'type'])
tags = list(set(selected_tags.tag))
tags.sort()
types = list(set(selected_tags.type))
types.sort()

# get metric dict
metric_dict = {}
for _type in types:
    metric_dict[_type] = {}
    metric_dict[_type]['p_10'] = []
    metric_dict[_type]['roc_auc'] = []
    metric_dict[_type]['pr_auc'] = []

# write scores
for i, tag in enumerate(tags):
    _type = selected_tags[selected_tags.tag==tag]['type'].item()
    metric_dict[_type]['p_10'].append(metrics['p_10'][i])
    metric_dict[_type]['roc_auc'].append(metrics['roc_auc'][i])
    metric_dict[_type]['pr_auc'].append(metrics['pr_auc'][i])

# average scores
for _type in types:
    p = np.mean(metric_dict[_type]['p_10'])
    roc = np.mean(metric_dict[_type]['roc_auc'])
    pr = np.mean(metric_dict[_type]['pr_auc'])
    print('%s: %.4f, %.4f, %.4f' % (_type, p, roc, pr))

