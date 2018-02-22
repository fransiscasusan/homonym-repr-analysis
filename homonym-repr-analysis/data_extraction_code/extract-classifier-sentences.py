import pickle
import xml.etree.cElementTree as ET
import codecs
import random

'''
This builds the datasets (train/dev/test) that get inputed into the classifier
'''

FILE = "../eurosense.v1.0.high-precision.xml"

with open('../top-senses-dict.pickle', 'rb') as handle:
    senses_dict = pickle.load(handle)

def preprocess(homonym, FILE, savepath):
    # i = j = 0

    all_possible_lemmas = senses_dict[homonym].keys()
    classes = {}
    for lemma in all_possible_lemmas:
        classes[lemma] = all_possible_lemmas.index(lemma)

    EN = []
    DE = []
    FR = []
    FI = []
    labels = []
    word_idxs = []

    en_id = de_id = fr_id = fi_id = 0
    homoynmexist = False

    sen = None
    current_eng_sentence = ''
    current_de_sentence = ''
    current_fi_sentence = ''
    current_fr_sentence = ''

    with open(FILE,'rb') as inputfile:

        for line in inputfile:

            if '<sentence id=' in line:
                en_id = de_id = fr_id = fi_id = 0
                homonymexist = False

            if '<text lang=\"de\"' in line:
                de_id = 1
                current_de_sentence = (ET.fromstring(line).text)

            if '<text lang=\"en\">' in line:
                en_id = 1
                current_eng_sentence = (ET.fromstring(line).text)

                if current_eng_sentence == None:
                    continue

                if homonym not in current_eng_sentence.split():
                    continue
                else:
                    homonymexist = True

            if '<text lang=\"fr\"' in line:
                fr_id = 1
                current_fr_sentence = (ET.fromstring(line).text)

            if '<text lang=\"fi\"' in line:
                fi_id = 1
                current_fi_sentence = (ET.fromstring(line).text)

            if '<annotation lang=\"en\"' in line and homonymexist and en_id and de_id and fr_id and fi_id:
                lemma_text = ET.fromstring(line).attrib.get("lemma")
                sense = ET.fromstring(line).text

                if sense in classes:
                    DE.append(current_de_sentence)
                    EN.append(current_eng_sentence)
                    FR.append(current_fr_sentence)
                    FI.append(current_fi_sentence)
                    labels.append(classes[sense])
                    word_idxs.append(current_eng_sentence.split().index(homonym))

    length = len(EN)
    inds = range(length)
    random.shuffle(inds)

    sep = int(8*length/12)
    sep2 = int(10*length/12)
    train_indexes = inds[:sep]
    dev_indexes = inds[sep:sep2]
    test_indexes = inds[sep2:]

    sentences_dict = {}
    word_idxs_dict = {}
    labels_dict = {}

    sentences_dict['train'] = [EN[i] for i in train_indexes]
    sentences_dict['dev'] = [EN[i] for i in dev_indexes]
    sentences_dict['test'] = [EN[i] for i in test_indexes]

    word_idxs_dict['train'] = [word_idxs[i] for i in train_indexes]
    word_idxs_dict['dev'] = [word_idxs[i] for i in dev_indexes]
    word_idxs_dict['test'] = [word_idxs[i] for i in test_indexes]

    labels_dict['train'] = [labels[i] for i in train_indexes]
    labels_dict['dev'] = [labels[i] for i in dev_indexes]
    labels_dict['test'] = [labels[i] for i in test_indexes]

    for data_type in ['train','dev','test']:
        f = codecs.open(savepath.format(data_type), 'w', 'utf-8')
        for idx, sentence in enumerate(sentences_dict[data_type]):
            f.write(str(word_idxs_dict[data_type][idx])+" "+str(labels_dict[data_type][idx])+" "+sentence+"\n")
        f.close()

    return classes

for homonym in senses_dict.keys():
    print homonym
    preprocess(homonym, FILE, homonym+'{}.txt')