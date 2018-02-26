import codecs

with codecs.open('words.txt', "r", "utf-8") as f:
	for line in f:
		wordlist = line.split()

PATH = "/data/sls/u/urop/fsusan/nmt-homo-analysis/data_lang_comparison/{}.txt"

def extractMajorityFromTraining(homonym):
	print(homonym)
	semdics = {}
	with codecs.open(PATH.format(homonym+'train'), "r", "utf-8") as corpus_file:
		for line in corpus_file:
			semindex = int(line.split()[1])
			semdics[semindex] = semdics.get(semindex, 0) + 1

	maxx = 0
	maxidx = 0
	for semindex in semdics:
		if semdics[semindex] > maxx:
			maxidx = semindex
			maxx = semdics[semindex]

	train_acc = maxx*1.0/sum(semdics.values())
	return maxidx, train_acc

def getAccuracy(homonym):
	maxidx, train_acc = extractMajorityFromTraining(homonym)

	with codecs.open(PATH.format(homonym+'dev'), "r", "utf-8") as corpus_file:
		semdics = {}
		for line in corpus_file:
			semindex = int(line.split()[1])
			semdics[semindex] = semdics.get(semindex, 0) + 1
		dev_acc = semdics[maxidx]*1.0/sum(semdics.values())

	with codecs.open(PATH.format(homonym+'test'), "r", "utf-8") as corpus_file:
		semdics = {}
		for line in corpus_file:
			semindex = int(line.split()[1])
			semdics[semindex] = semdics.get(semindex, 0) + 1
		test_acc = semdics[maxidx]*1.0/sum(semdics.values())

	print('Homoynm: '+ homonym )
	print('Train Accuracy: {:.6f}'.format( train_acc))
	print('Dev Accuracy: {:.6f}'.format( dev_acc))
	print('Test Accuracy: {:.6f}\n'.format( test_acc))

for homonym in wordlist:
	getAccuracy(homonym)