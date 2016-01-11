import nltk
from nltk.corpus import brown
from nltk.probability import *
import numpy.random as npr
import cPickle
target_word_set = [u'whoever', u'over', u'via', u'through', u'yet',u'before', u'whose',  u'how', u'should', u'to', u'under', u'might', u'ought', u'do', u'them',  u'around', u'outside', u'every', u'during', u'nor', u'like', u'this', u'either', u'though', u'each', u'because',  u'some', u'past', u'beyond', u'out', u'nether', u'what', u'for', u'since', u'below', u'per', u'ether',  u'behind', u'above', u'between', u'neither', u'across', u'who', u'however', u'although', u'along', u'by', u"'round", u'on', u'about', u'of', u'could', u'according', u'against', u'onto', u'or',u'among', u'besides', u'into', u'within', u'one', u'down',  u'another', u'throughout', u"'nother", u"'till", u'from', u'would',  u'next', u'whom', u'until', u'that', u'concerning', u'but', u'with', u'than', u'those', u'must', u'unlike', u'whether', u'inside', u'up', u'will', u'while', u'can', u'toward', u'and', u'then', u'an', u'as', u'at', u'in', u'any', u'if', u'these', u'no', u'rather', u'beside', u'till', u'towards', u'unless', u'shall', u'may', u'after', u'upon', u'a', u'without', u'so', u'the', u'once', u'which', u'yourselves', u'its', u'herself', u'ours', u'he', u'his', u'somebody', u'myself', u'they', u'yourself', u'him', u'hisself', u'someone', u'our', u'ourselves',  u'everything', u'she', u'we', u'hers', u'her', u'anything', u'everyone', u'your', u'everybody', u'anyone', u'their', u'themselves',  u'himself', u'nobody', u'mine', u'me', u'none', u'us', u'theirs', u'my', u'something', u'it', u'itself', u'you', u'oneself', u'nothing', u'i', u'anybody', u'yours']


words = FreqDist()

for sentence in brown.sents():
    for word in sentence:
        words[word.lower()] += 1

word_list_size = 1000

lst = words.most_common(word_list_size)
w = [v[0] for v in lst]
selected_target = [v for v in target_word_set if v in w]

tmp = brown.words()

word2intdict = {}
for i in range(len(w)):
    word2intdict[w[i]] = i + 1

for v in tmp:
    if not v in w:
        word2intdict[v.lower()] = 0

word2intdict['X'] = word_list_size + 1 #missing words
word2intdict['B'] = word_list_size + 2 #begin
word2intdict['E'] = word_list_size + 3 #end


target2intdict = {}
for i in range(len(selected_target)):
    target2intdict[selected_target[i]] = i + 1


def prepare_0(data, word2intdict, windows_size = 3):
    result_x = []
    result_y = []
    for sent in data:
        for k in range(windows_size / 2):
            sent = ['B'] + sent + ['E']
        length = len(sent)
        for i in range(windows_size / 2, length - windows_size / 2):
            if (sent[i] in selected_target):
                concat = []
                for j in range(-(windows_size / 2), windows_size / 2 + 1):
                    one_hot = [0 for v in range(word_list_size + 4)]
                    if j == 0 :
                        one_hot[word2intdict[sent[i + j].lower()]] = 1
                    else :
                        one_hot[word2intdict['X']] = 1
                    concat = concat + one_hot
                result_x.append(concat)
                one_hot = [0 for v in range(len(selected_target) + 1)]
                one_hot[target2intdict[sent[i].lower()]] = 1
                result_y.append(one_hot)
    return result_x, result_y

def data_api(spilt_rate):
    raw_sent = brown.sents()
    partial_data = raw_sent[:int(0.1*len(raw_sent))]

    data_x, data_y = prepare_0(partial_data, word2intdict)

    print 'len data_x', len(data_x), len(data_y)

    train_inds = npr.choice(range(len(data_x)), size = int((1 - spilt_rate) * len(data_x)), replace = False)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    print 'len train_inds', len(train_inds), len(data_x)
    for i in range(len(data_x)):
        if i in train_inds:
        	#print 'trn', i
            X_train.append(data_x[i])
            Y_train.append(data_y[i])
        else :
        	#print 'tst', i
            X_test.append(data_x[i])
            Y_test.append(data_y[i])
    print 'len X_train', len(X_train), len(X_test)
    return (X_train, Y_train), (X_test, Y_test)

(X_train, Y_train), (X_test, Y_test) = data_api(0.2)
f = open('data.pkl', 'w')
cPickle.dump(((X_train, Y_train), (X_test, Y_test)), f)
f.close()
print len(X_train), len(X_test)
