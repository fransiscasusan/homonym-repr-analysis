import codecs

with codecs.open('/usr/users/fsusan/homonym-repr-analysis/src/words.txt', "r", "utf-8") as f:
	for line in f:
		wordlist = line.split()

def findMax(homonym):
	with codecs.open(homonym+'training.txt', "r", "utf-8") as f:
		train_accs = []
		dev_accs = []
		i = 0
		for line in f:
			if 'Accuracy is' in line:
				acc = float(line.split()[2])
				if i % 2 == 0:
					train_accs.append(acc)
				else:
					dev_accs.append(acc)
				i += 1
		dev_acc = max(dev_accs)
		max_idx = dev_accs.index(dev_acc)
		train_acc = train_accs[max_idx]
		epoch_num = max_idx+1
		return (train_acc, dev_acc, epoch_num)

for word in wordlist:
	res = findMax(word)
	print res