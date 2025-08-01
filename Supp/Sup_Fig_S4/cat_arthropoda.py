import pandas as pd
import glob

targets = glob.glob("Arthropoda*")
tmp = []
for file_name in targets:
	df = pd.read_csv(file_name, sep='\t')
	tmp.append(df)
df = pd.concat(tmp)

df.to_csv("Arthropoda.subsample.k-mer_dist.txt", sep='\t', index=False)