import pandas as pd

def load_scores(file_name, method):
	df = pd.read_csv(file_name, sep=',')
	df['method'] = method

	return df

deepcoi = load_scores("DeepCOI/DeepCOI-excluded.scores.csv", 'DeepCOI')
rdp = load_scores("RDP/RDP.scores.excluded.csv", 'RDP')
blast = load_scores("BLAST/BLAST.scores.excluded.csv", 'BLASTn')

df = pd.concat([deepcoi, rdp, blast])
df.to_csv("excluded.csv", index=False)