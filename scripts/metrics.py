import os
import itertools as it
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
RANKS = ['phylum', 'class', 'order', 'family', 'genus', 'species'][1:]

####
## For
## AUROC and AUPR
####

def get_taxa_info(names):
	num_taxa = {
		rank: np.char.startswith(names, f"{rank[0]}__").sum()
		for rank in RANKS
	}
	end_pos = list(it.accumulate(num_taxa.values()))
	start_pos = [0] + end_pos[:-1]

	taxa_info = {}
	for rank, spos, epos in zip(RANKS, start_pos, end_pos):
		taxa_info[rank] = (spos, epos)

	return taxa_info

def get_trues(sid_list, meta):
	CM = meta['DAG'][()].toarray()
	labels = []
	for sid in sid_list:
		label = sid.split('|')[-1]
		try:
			species = f"s__{label}"
			idx = np.where(meta['names'] == species)[0][0]
		except:
			genus = f"g__{label}"
			idx = np.where(meta['names'] == genus)[0][0]

		label = CM[:, idx]
		labels.append(label)

	return np.array(labels)

def load_predictions(prefix, meta, targets=None):
	y_pred = np.load(f"{prefix}.test.npy", allow_pickle=True)[()]
	if targets is not None:
		sid_map = {sid.split(' |')[0]:sid for sid in y_pred.keys()}
		y_pred = {sid_map[sid]:y_pred[sid_map[sid]] for sid in targets}
	y_true = get_trues(y_pred.keys(), meta)
	y_pred = np.array(list(y_pred.values()))

	return y_true, y_pred

def make_micro_data(y_true, y_pred):
	preds_ids = y_pred.argmax(axis=1) + 1
	y_pred = y_pred.max(axis=1)

	total = y_true.shape[0]
	true_ids = np.where(y_true == 1)[1] + 1
	true_ids = np.concatenate([true_ids, np.zeros(total - true_ids.shape[0])])
	y_true = (true_ids == preds_ids).astype(float)

	return y_true, y_pred

def balancing(indexes, sub_true, sub_pred, y_true, y_pred):
	labels, counts = np.unique(sub_true, return_counts=True)
	cnt_map = {label:count for label, count in zip(labels, counts)}

	n_positive = cnt_map[1.0] if 1.0 in cnt_map else 0
	n_negative = cnt_map[0.0] if 0.0 in cnt_map else 0
	n_sample = max([0, n_positive - n_negative])
	n_sample = min([y_true.shape[0] - sub_true.shape[0], n_sample])
	if n_sample != 0:
		probs = np.ones(y_true.shape[0])
		probs[indexes] = 0
		probs = probs / probs.sum()

		ext_indexes = np.random.choice(np.arange(y_true.shape[0]), n_sample, p=probs)
		ext_true = y_true[ext_indexes]
		ext_pred = y_pred[ext_indexes]

		sub_true = np.concatenate([sub_true, ext_true])
		sub_pred = np.concatenate([sub_pred, ext_pred])

	return sub_true, sub_pred

def make_macro_data(y_true, y_pred, n_classes):
	pred_ids = y_pred.argmax(axis=1) + 1
	total = y_true.shape[0]
	true_ids = np.where(y_true == 1)[1] + 1
	true_ids = np.concatenate([true_ids, np.zeros(total - true_ids.shape[0])])

	sub_true, sub_pred = [], []
	for idx in range(1, n_classes + 1):
		tmp_true = np.where(true_ids == idx)[0]
		tmp_pred = np.where(pred_ids == idx)[0]

		indexes = np.unique(np.concatenate([tmp_true, tmp_pred]))

		tmp_true = y_true[indexes, idx - 1]
		tmp_pred = y_pred[indexes, idx - 1]

		tmp_true, tmp_pred = balancing(indexes, tmp_true, tmp_pred, y_true[:, idx - 1], y_pred[:, idx - 1])

		sub_true.append(tmp_true)
		sub_pred.append(tmp_pred)

	return sub_true, sub_pred

def prepare(y_true, y_pred, classes, metric='roc', return_th=False):
	a, b, th = dict(), dict(), dict()
	c = dict()
	x, y = (a, b) if metric == 'roc' else (b, a)
	curve_func = roc_curve if metric == 'roc' else precision_recall_curve

	micro_true, micro_pred = make_micro_data(y_true, y_pred)
	a['micro'], b['micro'], th['micro'] = curve_func(micro_true, micro_pred)
	c['micro'] = auc(x['micro'], y['micro'])

	if metric == 'roc':
		macro_true, macro_pred = make_macro_data(y_true, y_pred, len(classes))
		for i in range(len(classes)):
			taxon = classes[i]
			if len(macro_true[i]) == 0:
				continue
			a[taxon], b[taxon], th[taxon] = curve_func(macro_true[i], macro_pred[i])
			c[taxon] = auc(x[taxon], y[taxon])
	else:
		for i in range(len(classes)):
			taxon = classes[i]
			a[taxon], b[taxon], th[taxon] = curve_func(y_true[:, i], y_pred[:, i])
			c[taxon] = auc(x[taxon], y[taxon])

	a_grid = np.linspace(0.0, 1.0, 1000)
	mean_b = np.zeros_like(a_grid)

	for taxon in classes:
		if taxon not in a or taxon not in b:
			continue
		if np.isnan(a[taxon]).any() or np.isnan(b[taxon]).any():
			continue
		mean_b += np.interp(a_grid, a[taxon], b[taxon])
	mean_b /= len(classes)

	a['macro'], b['macro'] = a_grid, mean_b
	try:
		c['macro'] = auc(x['macro'], y['macro'])
	except:
		c['macro'] = auc(y['macro'], x['macro'])

	if return_th:
		all_th = np.unique(np.concatenate([th[taxon] for taxon in classes if taxon in th]))
		th_a = np.zeros_like(all_th)

		for taxon in classes:
			if taxon not in th:
				continue
			th_a += np.interp(all_th, th[taxon], a[taxon][::-1])
		th_a /= len(classes)

		th['macro'] = all_th
		th['micro'] = th['micro']
		th_a = {'micro': a['micro'], 'macro': th_a}

	if return_th:
		return a, b, c, (th, th_a)
	return a, b, c

def get_curves(meta_file, models, desc=None, targets=None):
	def make_rank_dict():
		return {rank:{} for rank in RANKS}

	fpr, tpr, roc_auc = make_rank_dict(), make_rank_dict(), make_rank_dict()
	precision, recall, pr_auc = make_rank_dict(), make_rank_dict(), make_rank_dict()
	th, th_fpr = make_rank_dict(), make_rank_dict()

	for model in tqdm(models, total=len(models), desc=desc):
		meta = np.load(meta_file, allow_pickle=True)
		taxa_info = get_taxa_info(meta['names'])
		
		y_true, y_pred = load_predictions(model, meta, targets)

		for rank in RANKS:
			spos, epos = taxa_info[rank]
			sub_true = y_true[:, spos:epos]
			sub_pred = y_pred[:, spos:epos]
			names = meta['names'][spos:epos]

			# ROC curve
			x, y, z, (a, b) = prepare(sub_true, sub_pred, names, metric='roc', return_th=True)
			fpr[rank][model], tpr[rank][model], roc_auc[rank][model] = x, y, z
			th[rank][model], th_fpr[rank][model] = a, b

			# PR curve
			x, y, z = prepare(sub_true, sub_pred, names, metric='pr')
			precision[rank][model], recall[rank][model], pr_auc[rank][model] = x, y, z

	return {
		"FPR": fpr, "TPR": tpr, "AUROC": roc_auc,
		"Precision": precision, "Recall": recall, "AUPR": pr_auc,
		"thresholds": th, "th_fpr": th_fpr
	}