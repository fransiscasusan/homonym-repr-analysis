import xml.etree.cElementTree as ET
import sys
import codecs
import pickle

LEMMA_COUNT = {}
senses = {}

i = 0

with open('../eurosense.v1.0.high-precision.xml','rb') as inputfile:

    for line in inputfile:

        if '<annotation lang=\"fr\"' in line:
            lemma_text = ET.fromstring(line).attrib.get("lemma")
            sense = ET.fromstring(line).text
            if lemma_text in senses:
                senses[lemma_text][sense] = senses[lemma_text].get(sense, 0) + 1
            else:
                senses[lemma_text] = {sense:1}
            LEMMA_COUNT[ lemma_text ] = LEMMA_COUNT.get(lemma_text, 0) + 1
        i += 1
        print(i)

top_annotated_words = []
for key, value in sorted(LEMMA_COUNT.iteritems(), key = lambda (k,v) : (-1*v,k))[:200]:
    top_annotated_words.append(key)

new_senses_dict = {}
for word in top_annotated_words:
    word_sense_dict = senses[word]
    newdict = {}
    for lemma in word_sense_dict:
        if word_sense_dict[lemma] >= 1 and word_sense_dict[lemma] >= 0.2 * LEMMA_COUNT[word]:
            newdict[lemma] = word_sense_dict[lemma]
    if len(newdict) > 1:
        new_senses_dict[word] = newdict

print("senses_dict\n",new_senses_dict)
print("\nlength\n",len(new_senses_dict))

with open('top-senses-dict-fr.pickle', 'wb') as handle:
    pickle.dump(new_senses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
