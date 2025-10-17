import sys
import os.path
import subprocess as sp
import multiprocessing as mp
from tqdm import tqdm

def check_db(phylum):
	exts = ['ndb', 'nhr', 'nin', 'njs', 'not', 'nsq', 'ntf', 'nto']
	flag = True
	for ext in exts:
		flag = flags and os.path.isfile(f"dbs/{phylum}.{ext}")

	return flag

N_PROCESSES = 2
N_THREADS = 20

cmds = []
phyla = ['Annelida', 'Arthropoda', 'Chordata', 'Cnidaria', 'Echinodermata', 'Mollusca', 'Nematoda', 'Platyhelminthes']
for phylum in phyla:
	if not check_db(phylum):
		print(f"{phylum} DB does NOT exist", file=sys.stderr)
		continue

	query = f'../../data/{phylum}.test.fasta'
	if not os.path.isfile(query):
		print(f"{query} does NOT exist", file=sys.stderr)
		continue

	cmd = ['blastn',
		'-db', f"dbs/{phylum}",
		'-num_threads', str(N_THREADS),
		'-outfmt', '6',
		'-evalue', '1.0e-10',
		'-out', f'm6/{phylum}.test.m6',
		'-query', query
	]

	cmds.append(cmd)

pool = mp.Pool(N_PROCESSES)
for _ in tqdm(pool.imap_unordered(sp.run, cmds)):
	pass
pool.close()
pool.join()
