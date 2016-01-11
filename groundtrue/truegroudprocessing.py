from nltk.tokenize import *
from nltk import *
from nltk.tag.hunpos import HunposTagger
import os
import nltk
import unicodedata


def extract_info_from_file(text_name):
    raw = open(text_name).readlines()

    word = []
    for l in raw:
        tmp = l.split(' ')
        for i in range(len(tmp)):
            if (tmp[i] == "*" or (tmp[i] == "." and tmp[i - 1] == "Mrs" or tmp[i - 1] == 'Mr') or tmp[i] == '\n'):
                continue
            boo = []
            boo.append(tmp[i].lower())
            if (i < len(tmp) - 1 and tmp[i + 1] == "*"):
                boo.append(1)
            else:
                boo.append(0)
            word.append(boo)
    text = []
    for w in word:
        #print w[0]
        #print str(w[0])
        if w[0].decode('unicode-escape') ==  w[0]:
            text.append(w[0])
        else:
            text.append(" ")
    #POS_word = pos_tag(text)
    #text = ' '.join(text)
    #text = nltk.word_tokenize(text)

    #print text
    #ht = HunposTagger('en_wsj.model')
    POS_word = pos_tag(text)
    #print POS_word
    return word, POS_word
    #print word
    #token = tokenize.word_tokenize(raw)
    #print (token[:1000])

word = extract_info_from_file('2.txt')
