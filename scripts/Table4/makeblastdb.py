import os
from Bio import SeqIO
import subprocess as sp

def collect_samples():
	sids = []
	for i in range(1, 5):
		parser = SeqIO.parse(f"../data/realworld/DS-PBBC{i}.fasta", format='fasta')
		for record in parser:
			sid = record.id.split('|')[0]
			sids.append(sid)
	return set(sids)

def filtering_samples(file_name, output, excluded):
	ostream = open(output, 'w')
	total, filtered = 0, 0
	parser = SeqIO.parse(file_name, format='fasta')
	for record in parser:
		total += 1
		if record.id in excluded:
			continue
		print(f'>{record.description}', file=ostream)
		print(record.seq, file=ostream)
		filtered += 1
	ostream.close()

	print(f"{file_name}: {total} seqs")
	print(f"{output}: {filtered} seqs")

excluded = collect_samples()

filtering_samples(
	"../data/Arthropoda.train.fasta",
	"Arthropoda_woDS-PBBC.train.fasta",
	excluded
)

os.makedirs('BLAST/db', exist_ok=True)
makeblastdb = [
	'makeblastdb',
	'-in', "Arthropoda_woDS-PBBC.train.fasta",
	'-out', 'BLAST/db/Arthropoda_woDS-PBBC',
	'-dbtype', 'nucl'
]
_ = sp.run(makeblastdb)