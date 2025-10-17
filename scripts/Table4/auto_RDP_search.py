import sys
import os
import subprocess as sp
import multiprocessing as mp

classifier = '../data/RDP/rdp_classifier_2.14/dist/classifier.jar'
if not os.path.isfile(classifier):
	print("RDP classifier does not installed", file=sys.stderr)
	exit(-1)

os.makedirs("RDP/raw", exist_ok=True)
for i in range(1, 2):
	trained = f'../data/realworld/rdp_woDS-PBBC/rRNAClassifier.properties'
	if not os.path.isfile(trained):
		print(f"trained RDP classifier ({trained}) does NOT exist", file=sys.stderr)
		continue

	query = f'../data/realworld/DS-PBBC{i}.fasta'
	if not os.path.isfile(query):
		print(f"query file ({query}) does NOT exist", file=sys.stderr)
		continue

	cmd = ['java', '-Xmx5g', '-jar',
		classifier,
		'-t', trained,
		'-o', f"RDP/raw/DS-PBBC{i}.rdp.txt",
		query
	]

	_ = sp.run(cmd)
