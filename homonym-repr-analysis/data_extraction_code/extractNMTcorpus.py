import xml.etree.cElementTree as ET
import sys
import codecs
import random

'''
This builds the datasets (train/dev/test) to train the OpenNMT system
'''

#1920130 + 1
n=10
i = 0

EN = []
DE = []
FR = []
FI = []

en_id = de_id = fr_id = fi_id = 0
sen = None
current_eng_sentence = ''
current_de_sentence = ''
current_fi_sentence = ''
current_fr_sentence = ''

with open('../eurosense.v1.0.high-precision.xml','rb') as inputfile:
  for line in inputfile:
    # if i>n:
    #     break
    if '<sentence id=' in line:
        i += 1
        print(i)
        en_id = de_id = fr_id = fi_id = 0
    if '<text lang=\"de\">' in line:
        de_id = 1
        current_de_sentence = ET.fromstring(line).text
        #DE.append(ET.fromstring(line).text)
    if '<text lang=\"en\">' in line:
        en_id = 1
        current_eng_sentence = ET.fromstring(line).text
        #EN.append(ET.fromstring(line).text)  
    if '<text lang=\"fi\">' in line:
        fi_id = 1
        current_fi_sentence = ET.fromstring(line).text
    if '<text lang=\"fr\">' in line:   
        fr_id = 1
        current_fr_sentence = ET.fromstring(line).text
        if fi_id == 1 and de_id == 1 and fr_id == 1 and en_id == 1:
            DE.append(current_de_sentence)
            EN.append(current_eng_sentence)
            FI.append(current_fi_sentence)
            FR.append(current_fr_sentence)

length = len(EN)
inds = range(length)
random.shuffle(inds)

sep0 = int(0.6*length)
sep1 = int(0.8*length)

train_indexes = inds[:sep0]
dev_indexes = inds[sep0:sep1]
test_indexes = inds[sep1:]

def createCorpus(indexes, data_name):
    corpusdata = {}
    corpusdata['src'] = ([EN[i] for i in indexes], 'en')
    corpusdata['tgt_DE'] = ([DE[i] for i in indexes], 'de')
    corpusdata['tgt_FR'] = ([FR[i] for i in indexes], 'fr')
    corpusdata['tgt_FI'] = ([FI[i] for i in indexes], 'fi')
    
    for lang in ['src', 'tgt_DE', 'tgt_FR', 'tgt_FI']:
        filename = 'nmt-'+data_name+'-'+corpusdata[lang][1]+'.txt'
        file = codecs.open(filename, 'w', 'utf-8')
        for item in corpusdata[lang][0]:
            file.write("%s\n" % item)
        file.close()

createCorpus(train_indexes, 'train')
createCorpus(dev_indexes, 'dev')
createCorpus(test_indexes, 'test')