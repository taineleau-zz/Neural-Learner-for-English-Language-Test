import math
import nltk
from nltk.corpus import brown
from nltk.util import *
from truegroudprocessing import *

target_set = ['AT', 'CC', 'CS', 'DO', 'DT', 'DTI', 'DTS', 'DTX', 'IN', 'MD', 'WPO', "WP$", 'WPS', 'WQL', 'WRG', 'WDT', 'PPO', 'PPS', 'PPSS', 'PPLS', 'PN', 'PP$', 'PP$$', 'PPL']

target_word_set = [u'whoever', u'over', u'via', u'through', u'yet',u'before', u'whose',  u'how', u'should', u'to', u'under', u'might', u'ought', u'do', u'them',  u'around', u'outside', u'every', u'during', u'nor', u'like', u'this', u'either', u'though', u'each', u'because',  u'some', u'past', u'beyond', u'out', u'nether', u'what', u'for', u'since', u'below', u'per', u'ether',  u'behind', u'above', u'between', u'neither', u'across', u'who', u'however', u'although', u'along', u'by', u"'round", u'on', u'about', u'of', u'could', u'according', u'against', u'onto', u'or',u'among', u'besides', u'into', u'within', u'one', u'down',  u'another', u'throughout', u"'nother", u"'till", u'from', u'would',  u'next', u'whom', u'until', u'that', u'concerning', u'but', u'with', u'than', u'those', u'must', u'unlike', u'whether', u'inside', u'up', u'will', u'while', u'can', u'toward', u'and', u'then', u'an', u'as', u'at', u'in', u'any', u'if', u'these', u'no', u'rather', u'beside', u'till', u'towards', u'unless', u'shall', u'may', u'after', u'upon', u'a', u'without', u'so', u'the', u'once', u'which', u'yourselves', u'its', u'herself', u'ours', u'he', u'his', u'somebody', u'myself', u'they', u'yourself', u'him', u'hisself', u'someone', u'our', u'ourselves',  u'everything', u'she', u'we', u'hers', u'her', u'anything', u'everyone', u'your', u'everybody', u'anyone', u'their', u'themselves',  u'himself', u'nobody', u'mine', u'me', u'none', u'us', u'theirs', u'my', u'something', u'it', u'itself', u'you', u'oneself', u'nothing', u'i', u'anybody', u'yours']

def get_data(text):
    data_list = []
    predict_word = []
    #print(text[1000])
    for w in text:
        t = {}
        t[0] = w[0].lower()
        if ( (w[1] in target_set and w[0] in target_word_set)):
            t[1] = 1
            predict_word.append(t[0])
        else:
            t[1] = 0
        data_list.append(t)
    #print(predict_word)
    predict_word = set(predict_word)
    return data_list

def ngram_baseline(text):
    ngs = ngrams(text, 2)
    cnt = 0
    """
    for t in ngs:
        print(t, )
        cnt = cnt + 1
        if (cnt > 1000):
            break
    """
    refine = []
    for (first, second) in ngs:
        if (second[1] == 1):
            #print(first[0], second[0], zip(first[0], second[0]))
            #tmp = (first[0], second[0])
            #print(tmp)
            #break
            refine.append((first[0], second[0]))
    cnt = 0
    """
    for t in refine:
        print(t)
        cnt = cnt + 1
        if (cnt > 1000):
            break
    #print(ngs)
    """
    cfdist = nltk.ConditionalFreqDist(refine)
    return cfdist

def get_feature_0(data):
    feature_set = []
    size = len(data)
    for i in range(size):
        tmp = {}
        if (data[i][1] == 1):
            if (i > 0):
                tmp['pre_words'] = data[i - 1][0]
        feature_set.append((tmp, data[i][0]))
    return feature_set

#
def get_feature_1(data, raw):
    feature_set = []
    size = len(data)
    for i in range(size):
        if (data[i][1] == 1):
            tmp = {}
            if (i > 0):
                tmp['pre_words'] = data[i - 1][0]
            if (i < size - 1):
                tmp['next_words_POS'] = raw[i + 1][1]
            feature_set.append((tmp, data[i][0]))

    return feature_set

#
def get_feature_2(data, raw):
    feature_set = []
    size = len(data)
    cnt = 0
    for i in range(size):
        if (data[i][1] == 1):
            tmp = {}
            if (i > 0):
                tmp['pre_words'] = data[i - 1][0]
            if (i < size - 1):
                tmp['next_words'] = raw[i + 1][0]
            if (i < size - 2):
                tmp['next_2_POS'] = raw[i + 2][1]
            if (i > 1):
                tmp['pre_-2_POS'] = raw[i - 2][1]
            #if (cnt < 1000):
            #    cnt = cnt + 1
            #    print tmp
            feature_set.append((tmp, data[i][0]))
    return feature_set



def classify(data_set):
    size = int(0.9 * len(data_set))
    print size
    train_set = data_set[:size]
    valid_set = data_set[size:]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm='gis',max_iter=10)
    print nltk.classify.accuracy(classifier, valid_set)


def make_true_data():
    word, tag_word = extract_info_from_file('2.txt')
    res = get_feature_2(word, tag_word)
    return res

def classify_true(data_set):
    valid_set = make_true_data()
    classifier = nltk.NaiveBayesClassifier.train(data_set)
    print nltk.classify.accuracy(classifier, valid_set)

raw = brown.tagged_words(categories="reviews")
data = get_data(raw)

def get_answer(method, key):
    ans = method[key]
    if (ans == {}):
        return None
    return ans.most_common()[0][0]


ngram_method = ngram_baseline(data)


#test = ['lost', 'become', 'faith', 'table', 'loved', 'simply', 'was', 'tested', 'patient', 'had', 'way']
#for w in test:
#    print(w, get_answer(ngram_method, w))


### feature_oriented
#classify(get_feature_0(data))
#classify(get_feature_1(data, raw))
#classify(get_feature_2(data, raw))
classify_true(get_feature_2(data, raw))



#pp = [w for w in predict if w not in target_word_set]
#print(pp)
#print(len(target_word_set))
#print(len(data))
