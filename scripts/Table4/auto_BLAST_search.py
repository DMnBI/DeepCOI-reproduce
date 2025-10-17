import os
import subprocess as sp
from tqdm import tqdm

N_THREADS = 20

os.makedirs('BLAST/m6', exist_ok=True)
for i in range(1, 2):
	cmd = ['blastn',
		'-db', f"BLAST/db/Arthropoda_woDS-PBBC",
		'-num_threads', str(N_THREADS),
		'-outfmt', '6',
		'-evalue', '1.0e-10',
		'-out', f'BLAST/m6/DS-PBBC{i}.raw.m6',
		'-query', f'../data/realworld/DS-PBBC{i}.fasta'
	]

	_ = sp.run(cmd)
