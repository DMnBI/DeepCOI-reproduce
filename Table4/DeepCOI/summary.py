import sys
import pandas as pd

def summary(suffix):
	dfs = []
	for i in range(1, 5):
		sample = f"DS-PBBC{i}"
		labels = pd.read_csv(f"../../data/realworld/labels/{sample}.txt", sep='\t', low_memory=False, index_col='processid')

		board = {}
		with open(f"raw/{sample}{suffix}") as f:
			for line in f:
				items = line.strip().split('\t')
				sid = items[0].split('|')[0]
				board[sid] = {}

				for idx in range(1, len(items), 3):
					rank, name, score = items[idx:idx+3]
					board[sid][f"{rank}_label"] = labels.loc[sid, f"{rank}_name"]
					board[sid][f"{rank}_pred"] = name
					board[sid][f"{rank}_score"] = score
		board = pd.DataFrame(board).T
		dfs.append(board)
	return pd.concat(dfs)

RANKS = ['class', 'order', 'family', 'genus', 'species']

deepcoi = summary(".pred.txt")
deepcoi.to_csv(f"DeepCOI.summary.csv", sep=',')
no_mcm = summary(".pred.no_mcm.txt")
no_mcm.to_csv(f"DeepCOI.no_mcm.summary.csv", sep=',')
