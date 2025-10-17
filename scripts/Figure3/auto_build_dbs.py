import sys
import os.path
import subprocess as sp
from tqdm import tqdm

cmds = []
phyla = [
	'Annelida',
	'Arthropoda',
	'Chordata',
	'Cnidaria',
	'Echinodermata',
	'Mollusca',
	'Nematoda',
	'Platyhelminthes'
]

for phylum in phyla:
	train_file = f'../../data/{phylum}.train.fasta'
	if not os.path.isfile(train_file):
		print(f"{train_file} does NOT exist", file=sys.stderr)
		continue

	cmd = ['makeblastdb',
		'-out', f"dbs/{phylum}",
		'-in', train_file,
		'-dbtype', 'nucl'
	]

	_ = sp.run(cmd)
