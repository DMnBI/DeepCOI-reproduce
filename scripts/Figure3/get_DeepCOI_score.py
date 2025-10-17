from metrics import *
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

def melt(table, metric):
    df = table.unstack().reset_index()
    df.columns = ['rank', 'model', metric]
    df['pivot'] = [f"{row['model']}.{row['rank']}" for _, row in df.iterrows()]

    return df[['pivot', metric]]

def call_func(args):
    meta_file, model, subset = args
    
    p = model.split('-')[-1]
    labels = pd.read_csv(f"../../data/testset_labels/{p}.test.labels.txt", sep=',')
    if subset == 'included':
            targets = labels.query("species == 'included'")['sid']
        elif subset == 'excluded':
            targets = labels.query("species == 'excluded'")['sid']
        else:
            targets = None

    curves = get_curves(meta_file, [model], targets=targets)
    return curves, model

def report_scores(subset=None):
    roc_tables, pr_tables = [], []
    RANKS = ["class", "order", "family", "genus", "species"]
    phyla = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']
    if subset == 'excluded':
        RANKS = RANKS[:-1]

    cmds = []
    for phylum in phyla:
        meta_file = f"../../data/meta/{phylum}.meta.npz"
        model =  f"DeepCOI-{phylum}"

        cmds.append((meta_file, model, subset))

    pool = mp.Pool(8)
    for curves, model in tqdm(pool.imap_unordered(call_func, cmds), total=len(cmds), desc='Table1'):
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
    df['phylum'] = df['pivot'].apply(lambda x: x.split('.')[0].split('-')[-1])
    df['rank'] = df['pivot'].apply(lambda x: x.split('.')[1])
    df = df[['phylum', 'rank', 'AUROC', 'AUPR']]

    output = 'DeepCOI.scores.csv'
    if targets is not None:
        output = f'DeepCOI-{subset}.scores.csv'
    df.to_csv(output, index=False)
    
report_scores(subset=None)
report_scores(subset='included')
report_scores(subset='excluded')