import itertools as it
import pandas as pd
import numpy as np
from Bio import SeqIO

M6_COLS = [
	'qseqid', 'sseqid', 'identity',
	'mismatch', 'length', 'gapopen',
	'qstart', 'qend', 'sstart', 'send',
	'evalue', 'bitscore'
]
RANKS = ['class', 'order', 'family', 'genus', 'species']

# Load DB labels
meta = np.load("../data/meta/Arthropoda.meta.npz", allow_pickle=True)
names = meta['names']
DAG = meta['DAG'][()].toarray()

parser = SeqIO.parse("Arthropoda_woDS-PBBC.train.fasta", format='fasta')
sid2sp = dict(tuple(record.description.split(' |')) for record in parser)

def load_m6(file_name):
	m6 = pd.read_csv(file_name, sep='\t', names=M6_COLS)
	m6 = m6[['qseqid', 'sseqid', 'identity']]
	m6 = m6.drop_duplicates(subset='qseqid', keep='first')

	tmp = {'sseqid': []}
	tmp.update({f"{rank}_pred": [] for rank in RANKS})

	for sseqid in m6['sseqid'].unique():
	    species = sid2sp[sseqid]
	    idx = np.where(names == f"s__{species}")[0][0]
	    lineage = DAG[:, idx].astype(bool)
	    preds = names[lineage]
	    
	    tmp['sseqid'].append(sseqid)
	    for rank, pred in zip(RANKS, preds):
	        tmp[f'{rank}_pred'].append(pred[3:])
	        
	preds = pd.DataFrame(tmp)

	m6 = pd.merge(left=m6, right=preds, left_on='sseqid', right_on='sseqid', how='inner')
	for rank in RANKS:
		m6[f'{rank}_score'] = m6['identity'] / 100
	return m6

def load_label(sample):
	df = pd.read_csv(f"../data/realworld/labels/{sample}.txt", sep='\t', index_col=0)
	df.columns = [col.replace('_name', '_label') for col in df.columns]

	parser = SeqIO.parse(f"../data/realworld/{sample}.fasta", format='fasta')
	sids = [record.id.split('|')[0] for record in parser]

	return df.loc[sids]

def load_data(sample):
	m6 = load_m6(f"BLAST/m6/{sample}.raw.m6")
	m6['processid'] = m6['qseqid'].apply(lambda x: x.split('|')[0])
	labels = load_label(sample)

	df = pd.merge(m6, labels, left_on='processid', right_on='processid', how='outer')
	df = df.set_index('processid')

	columns = []
	for rank in RANKS:
		columns += [f"{rank}_label", f"{rank}_pred", f"{rank}_score"]

	return df[columns]

dfs = []
for i in range(1, 5):
	df = load_data(f"DS-PBBC{i}")
#	df.to_csv(f"DS-PBBC{i}.rdp.csv", sep=',', index=True, header=True)
	dfs.append(df)
df = pd.concat(dfs)

df.to_csv("BLAST/DS-PBBC.blast.csv", sep=',', index=True, header=True)
