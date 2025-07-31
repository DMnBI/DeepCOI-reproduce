from metrics import *
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

import os
os.chdir("../Sup_Table_S2/")

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
#phyla = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']
phyla = ['Annelida']
cmds = []
for phylum in phyla:
    meta_file = f"../../data/meta/{phylum}.meta.npz"
    for itermediate in ['no_network', '1D-CNN', 'GRU', '1D-CNN_GRU']:
        for pooler in ['mean', 'max', 'cls_token', 'convolutional_attention']:
            model = f"{itermediate}/{pooler}.{phylum}"
            cmds.append((meta_file, model))

pool = mp.Pool(4)
for curves, model in tqdm(pool.imap_unordered(call_func, cmds), total=len(cmds), desc='Sup_Table_S2'):
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

roc_table.to_csv("AUROC.csv", sep=',')
pr_table.to_csv("AUPR.csv", sep=',')
