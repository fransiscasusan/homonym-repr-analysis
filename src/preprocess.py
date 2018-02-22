import pickle
import xml.etree.cElementTree as ET
import codecs
import random

'''
This file gets the mapping of each lemma to its class index used in the classifier
'''

with open('/data/sls/scratch/fsusan/EuroSense/top-senses-dict.pickle', 'rb') as handle:
    senses_dict = pickle.load(handle)

def getLabelClasses(homonym):
    # i = j = 0

    all_possible_lemmas = senses_dict[homonym].keys()
    classes = {}
    for lemma in all_possible_lemmas:
        classes[lemma] = all_possible_lemmas.index(lemma)

    return classes