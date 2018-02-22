import torch
import torch.utils.data as data
import gzip
import numpy as np
import codecs
import onmt
import preprocess

PATH = "/data/sls/u/urop/fsusan/nmt-homo-analysis/data_lang_comparison/{}.txt"
FILE = "/data/sls/scratch/fsusan/EuroSense/eurosense.v1.0.high-precision.xml"

'''
input: text, homonym
output:
    self.semantics = [sem_1, sem_2] 
    self.data = {sentence:'', semindex: int, wordindex: int}
'''

class FullHomonymDataset(data.Dataset):

    def __init__(self, name, homonym, opt, dummy_opt, classes):
        self.name = name
        self.homonym = homonym
        self.dataset = []

        self.classes = classes
        self.processLines(PATH.format(homonym+name), opt, dummy_opt)

    def processLines(self, path_to_file, opt, dummy_opt):

        opt.cuda = opt.gpu > -1

        if opt.cuda:
            torch.cuda.set_device(opt.gpu)

        translator = onmt.Translator(opt, dummy_opt.__dict__)

        words_idx = []
        sems_idx = []
        with codecs.open(path_to_file, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                wordindex, semindex = int(line.split()[0]), int(line.split()[1])
                words_idx.append(wordindex)
                sems_idx.append(semindex)

        data = onmt.IO.ONMTDataset(path_to_file, None, translator.fields, None)
        train_data = onmt.IO.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            shuffle=False)

        word_encodings = []
        for i, batch in enumerate(train_data):
            # print batch.__dict__['src']
            word_idx = words_idx[i]
            encodings = translator.encode(batch, data)[word_idx]

            sample = {'x':encodings.data, 'y':sems_idx[i]}
            self.dataset.append(sample)
            #print(encodings)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def getAll(self):
        return self.dataset

    def getRepresentationLength(self):
        replength = int(self.dataset[0]['x'].size()[1])
        return replength

def load_dataset(homonym, opt, dummy_opt):
    print("\nLoading data...")

    # savepath = "/../data/"+homonym+'{}.txt'
    classes = preprocess.getLabelClasses(homonym)

    train_data = FullHomonymDataset('train', homonym, opt, dummy_opt, classes)
    dev_data = FullHomonymDataset('dev', homonym, opt, dummy_opt, classes)
    test_data = FullHomonymDataset('test', homonym, opt, dummy_opt, classes)
    replength = train_data.getRepresentationLength()

    return train_data, dev_data, test_data, replength, classes
