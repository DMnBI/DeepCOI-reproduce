import pandas as pd
RANKS = ['class', 'order', 'family', 'genus', 'species']

def load_RDP(file_name):
	f = open(file_name, 'r')
	tmp = {}
	for line in f:
		items = line.strip().split('\t')
		sid = items[0].split('|')[0]

		tmp[sid] = {}
		for idx in range(2, len(items), 3):
			pred, rank, score = items[idx:idx+3]
			if rank == 'rootrank':
				continue
			tmp[sid][f"{rank}_pred"] = pred
			tmp[sid][f"{rank}_score"] = score
	f.close()

	return pd.DataFrame(tmp).T

def load_label(file_name):
	df = pd.read_csv(file_name, sep='\t', index_col=0)
	df.columns = [col.replace('_name', '_label') for col in df.columns]

	return df

def load_data(sample):
	preds = load_RDP(f"raw/{sample}.rdp.txt")
	labels = load_label(f"../../data/realworld/labels/{sample}.txt")

	print(preds.shape, labels.shape)
	df = pd.merge(labels, preds, left_index=True, right_index=True, how='inner')
	print(df.shape)

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

df.to_csv("DS-PBBC.rdp.csv", sep=',', index=True, header=True)
