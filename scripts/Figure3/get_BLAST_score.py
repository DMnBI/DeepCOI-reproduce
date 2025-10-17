import sys

from Bio import SeqIO
from metrics import *

import numpy as np
import pandas as pd
import itertools as it
import pickle as pkl

from datetime import datetime
import pytz

def log(msg):
	ctime = datetime.now(pytz.timezone('Asia/Seoul'))
	ts = ctime.strftime("%Y-%m-%d %H:%M:%S")

	print(f"[{ts}] {msg}", flush=True)

def get_db_info(db):
	parser = SeqIO.parse(db, format='fasta')

	db_info = {}
	for record in parser:
		sid, species = record.description.split(' |')
		db_info[sid] = species

	return db_info

def load_m6(file_name, query, targets=None):
	def get_qinfo(query, targets=None):
		parser = SeqIO.parse(query, format='fasta')

		return {
			record.id:
			{'length':len(record.seq), 'taxon': record.description.split(' |')[-1]} 
			for record in parser if targets is None or record.id in targets.values
		}

	qinfo = get_qinfo(query, targets=targets)

	m6 = pd.read_csv(file_name, sep='\t', header=None)
	m6.columns = ['qseqid', 'sseqid', 'identity', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
	m6 = m6.query("qseqid in @qinfo.keys()")
	m6['qlen'] = [qinfo[sid]['length'] for sid in m6['qseqid']]
	m6['qcov'] = (m6['qend'] - m6['qstart'] + 1) / m6['qlen']

	return m6.query("qcov >= 0.85"), qinfo

def get_trues(qinfo, names, DAG):
	labels = []
	for sid, info in qinfo.items():
		label = info['taxon']
		try:
			idx = np.where(names == f"s__{label}")[0][0]
		except:
			idx = np.where(names == f"g__{label}")[0][0]

		label = DAG[:, idx]
		labels.append(label)

	return np.array(labels)

def load_preds(m6, qinfo, db_info, names, DAG):
	m6['species'] = [db_info[sid] for sid in m6['sseqid']]
	besthits = m6.groupby('qseqid').apply(lambda x: x.drop_duplicates(subset=['species'], keep='first')).reset_index(drop=True)

	y_pred = []
	for query in qinfo.keys():
		subdf = besthits.query('qseqid == @query')
		if subdf.shape[0] == 0:
			probs = np.zeros_like(DAG[:, 0])
		else:
			tmp = []
			for _, row in subdf.iterrows():
				idx = np.where(names == f"s__{row['species']}")[0][0]
				probs = DAG[:, idx] * (row['identity'] / 100)
				tmp.append(probs)

			probs = np.array(tmp).max(axis=0)
		y_pred.append(probs)

	y_pred = np.stack(y_pred)
	y_true = get_trues(qinfo, names, DAG)

	return y_true, y_pred

def melt(table, metric):
	df = table.unstack().reset_index()
	df.columns = ['rank', 'phylum', metric]
	df['pivot'] = [f"{row['phylum']}.{row['rank']}" for _, row in df.iterrows()]

	return df[['pivot', metric]]

def make_dict(PHYLA):
	return {phylum:dict() for phylum in PHYLA}

def report_scores(subset=None):
	RANKS = ['class', 'order', 'family', 'genus', 'species']
	PHYLA = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']

	fpr, tpr, roc_auc = make_dict(RANKS), make_dict(RANKS), make_dict(RANKS)
	precision, recall, pr_auc = make_dict(RANKS), make_dict(RANKS), make_dict(RANKS)

	for phylum in PHYLA:
		log(f"{phylum} START.")
		meta = np.load(f"../../data/meta/{phylum}.meta.npz", allow_pickle=True)
		names = meta['names']
		DAG = meta['DAG'][()].toarray()

		taxa_info = get_taxa_info(names)
		db_info = get_db_info(f"../../data/{phylum}.train.fasta")

		labels = pd.read_csv(f"../../data/testset_labels/{phylum}.test.labels.txt", sep=',')
		if subset == 'included':
			targets = labels.query("species == 'included'")['sid']
		elif subset == 'excluded':
			targets = labels.query("species == 'excluded'")['sid']
		else:
			targets = None

		preds, qinfo = load_m6(f"m6/{phylum}.test.m6", f"../../data/{phylum}.test.fasta", targets=targets)
		y_true, y_pred = load_preds(preds, qinfo, db_info, names, DAG)

		for rank in RANKS:
			spos, epos = taxa_info[rank]
			sub_true = y_true[:, spos:epos]
			sub_pred = y_pred[:, spos:epos]
			sub_names = meta['names'][spos:epos]

			fpr[rank][phylum], tpr[rank][phylum], roc_auc[rank][phylum], (_, _) = prepare(sub_true, sub_pred, sub_names, metric='roc', return_th=True)
			precision[rank][phylum], recall[rank][phylum], pr_auc[rank][phylum] = prepare(sub_true, sub_pred, sub_names, metric='pr')

		log(f"{phylum} DONE.")

	blast_roc_table = pd.DataFrame({rank:{phylum:roc_auc[rank][phylum]['macro'] for phylum in PHYLA} for rank in RANKS})
	blast_pr_table = pd.DataFrame({rank:{phylum:pr_auc[rank][phylum]['macro'] for phylum in PHYLA} for rank in RANKS})

	blast_roc_table = melt(blast_roc_table, 'AUROC')
	blast_pr_table = melt(blast_pr_table, 'AUPR')

	df = pd.merge(left=blast_roc_table, right=blast_pr_table, left_on='pivot', right_on='pivot')
	df['phylum'] = df['pivot'].apply(lambda x: x.split('.')[0])
	df['rank'] = df['pivot'].apply(lambda x: x.split('.')[1])
	df = df[['phylum', 'rank', 'AUROC', 'AUPR']]

	output = 'BLAST.scores.csv'
	if targets is not None:
		output = f'BLAST.scores.{subset}.csv'
	df.to_csv(output, index=False)

report_scores(subset=None)
report_scores(subset='included')
report_scores(subset='excluded')