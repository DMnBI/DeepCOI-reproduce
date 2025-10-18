import sys
import os.path
import subprocess as sp
import multiprocessing as mp

N_PROCESSES = 9

phyla = [
	'Annelida',
	'Arthropoda',
	'Chordata',
	'Cnidaria',
	'Echinodermata',
	'Mollusca',
	'Nematoda',
	'Platyhelminthes',
	'outgroups'
]

classifier = '../../data/RDP/rdp_classifier_2.14/dist/classifier.jar'
if not os.path.isfile(classifier):
	print("RDP classifier does not installed", file=sys.stderr)
	exit(-1)

trained = '../../data/RDP/trained/phylum/rRNAClassifier.properties'
if not os.path.isfile(trained):
	print(f"trained RDP classifier ({trained}) does NOT exist", file=sys.stderr)
	exit(-1)

cmds = []
for phylum in phyla:
	query = f'../../data/{phylum}.test.fasta'
	if not os.path.isfile(query):
		print(f"query file ({query}) does NOT exist", file=sys.stderr)
		continue

	cmd = ['java', '-Xmx5g', '-jar',
		classifier,
		'-t', trained,
		'-o', f"{phylum}.rdp.test.txt",
		query
	]

	cmds.append(cmd)

pool = mp.Pool(N_PROCESSES)
for _ in pool.map(sp.run, cmds):
	pass
pool.close()
pool.join()
