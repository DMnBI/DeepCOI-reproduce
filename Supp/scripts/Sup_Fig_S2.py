from metrics import *
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

import os
os.chdir("../Sup_Fig_S2")

RANKS = ["class", "order", "family", "genus", "species"]

def melt(table, metric):
    df = table.unstack().reset_index()
    df.columns = ['rank', 'model', metric]
    df['pivot'] = [f"{row['model']}.{row['rank']}" for _, row in df.iterrows()]

    return df[['pivot', metric]]

def call_func(args):
    meta_file, model = args

    curves = get_curves(meta_file, [model])
    return curves, model

roc_tables, pr_tables = [], []
phyla = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']
cmds = []
for phylum in phyla:
    meta_file = f"../../data/meta/{phylum}.meta.npz"
    models = [f"t6-320.{phylum}", f"t12-480.{phylum}"]
    names = models
    
    for model in models:
        cmds.append((meta_file, model))

pool = mp.Pool(4)
for curves, model in tqdm(pool.imap_unordered(call_func, cmds), total=len(cmds), desc='Sup_Fig_S2'):
    auroc = curves['AUROC']
    aupr = curves['AUPR']

    roc_tables.append(
        pd.DataFrame({
            rank: {model:auroc[rank][model]['macro']}
            for rank in RANKS
        })
    )

    pr_tables.append(
        pd.DataFrame({
            rank: {model:aupr[rank][model]['macro']}
            for rank in RANKS
        })
    )
    
roc_table = pd.concat(roc_tables)
pr_table = pd.concat(pr_tables)

roc_table = melt(roc_table, 'AUROC')
pr_table = melt(pr_table, 'AUPR')

df = pd.merge(left=roc_table, right=pr_table, left_on='pivot', right_on='pivot')
df['base'] = df['pivot'].apply(lambda x: x.split('.')[0])
df['phylum'] = df['pivot'].apply(lambda x: x.split('.')[1])
df['rank'] = df['pivot'].apply(lambda x: x.split('.')[2])
df = df[['base', 'phylum', 'rank', 'AUROC', 'AUPR']]

df.to_csv("Sup_Fig_S2.data.csv", index=False)