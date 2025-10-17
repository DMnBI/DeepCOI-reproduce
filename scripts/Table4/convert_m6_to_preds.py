import sys
sys.path.append("/home/hjgwak/scripts/modules")
import SeqModules as sm
import pandas as pd
import numpy as np

RANKS = ['class', 'order', 'family', 'genus', 'species']

def get_qinfo(query, targets=None):
	parser = sm.SeqParser(query, fmt='fasta')
	seqs = parser.read()

	return {
		seq['id']:
		{'length':len(seq['seq']), 'taxon': seq['header'].split(' |')[-1]} 
		for seq in seqs if targets is None or seq['id'] in targets.values
	}

def load_m6(file_name, query, targets=None):
	qinfo = get_qinfo(query)

	m6 = pd.read_csv(file_name, sep='\t', header=None)
	m6.columns = ['qseqid', 'sseqid', 'identity', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
#	m6 = m6.query("qseqid in @qinfo.keys()")
	m6['qlen'] = [qinfo[sid]['length'] for sid in m6['qseqid']]
	m6['qcov'] = (m6['qend'] - m6['qstart'] + 1) / m6['qlen']

	return m6.query("qcov >= 0.85"), qinfo

PHYLA = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']
for phylum in PHYLA:
	print(phylum)
	m6, qinfo = load_m6(f"../Figure3/BLAST/m6/{phylum}.test.m6", f"../data/{phylum}.test.fasta")
	db_info = get_qinfo(f"../data/{phylum}.train.fasta")
	
	meta = np.load(f"../data/meta/{phylum}.meta.npz", allow_pickle=True)
	names = meta['names']
	DAG = meta['DAG'][()].toarray()

	besthits = m6.drop_duplicates(subset=['qseqid'], keep='first')

	tmp = {'sid': [], 'rank': [], 'pred': [], 'score': []}
	for _, row in besthits.iterrows():
		tmp['sid'] += [row['qseqid']] * len(RANKS)
		
		hit = db_info[row['sseqid']]['taxon']
		idx = np.where(names == f"s__{hit}")[0][0]
		labels = DAG[:, idx].astype(bool)

		assign = names[labels]
		for rank, pred in zip(RANKS, assign):
			tmp['rank'].append(rank)
			tmp['pred'].append(pred[3:])
			tmp['score'].append(row['identity'])

	unaligned = [sid for sid in qinfo if sid not in besthits['qseqid'].values]
	for sid in unaligned:
		tmp['sid'] += [sid] * len(RANKS)
		tmp['rank'] += RANKS
		tmp['pred'] += ['unclassified'] * len(RANKS)
		tmp['score'] += [0.0] * len(RANKS)

	df = pd.DataFrame(tmp)
	df.to_csv(f"../invest/FPR/{phylum}.blast.csv", sep=',', index=False)