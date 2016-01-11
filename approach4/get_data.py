import nltk
from nltk.corpus import brown
from nltk.util import *
import numpy.random as npr
from nltk import FreqDist

target_set = ['AT', 'CC', 'CS', 'DO', 'DT', 'DTI', 'DTS', 'DTX', 'IN', 'MD', 'WPO', "WP$", 'WPS', 'WQL', 'WRG', 'WDT', 'PPO', 'PPS', 'PPSS', 'PPLS', 'PN', 'PP$', 'PP$$', 'PPL']

#target_word_set = [u'yourselves', u'its', u'herself', u'ours', u'he', u'his', u'somebody', u'myself', u'they', u'yourself', u'him', u'hisself', u'someone', u'our', u'ourselves',  u'everything', u'she', u'we', u'hers', u'her', u'anything', u'everyone', u'your', u'everybody', u'anyone', u'their', u'themselves',  u'himself', u'nobody', u'mine', u'me', u'none', u'us', u'theirs', u'my', u'something', u'it', u'itself', u'you', u'oneself', u'nothing', u'i', u'anybody', u'yours']

target_word_set = [u'whoever', u'over', u'via', u'through', u'yet',u'before', u'whose',  u'how', u'should', u'to', u'under', u'might', u'ought', u'do', u'them',  u'around', u'outside', u'every', u'during',  u'like', u'this', u'either', u'each', u'because',  u'some', u'past', u'beyond', u'out', u'what', u'for', u'since', u'below', u'per', u'ether',  u'behind', u'above', u'between', u'neither', u'across', u'who', u'however', u'although', u'along', u'by',  u'on', u'about', u'of', u'could', u'according', u'against', u'or',u'among', u'besides',  u'within', u'one', u'down',  u'another', u'throughout', u'from', u'would',   u'whom', u'until', u'that', u'concerning', u'but', u'with', u'than', u'those', u'must', u'unlike', u'whether', u'inside', u'up', u'will', u'while', u'can', u'toward', u'and', u'then', u'an', u'as', u'at', u'in', u'any', u'if', u'these', u'no', u'rather', u'beside', u'till',  u'unless', u'shall', u'may', u'after', u'a', u'without', u'so', u'the', u'once', u'which', u'yourselves', u'its', u'herself',  u'somebody', u'myself', u'they', u'yourself', u'him', u'hisself', u'someone', u'our', u'ourselves',  u'everything',  u'hers', u'her', u'anything', u'everyone',  u'everybody', u'anyone', u'themselves',  u'himself', u'nobody', u'mine', u'none',u'something', u'it', u'itself',  u'oneself', u'nothing',  u'anybody', 'when']

range_len = len(target_word_set)
assert(range_len == len(set(target_word_set)))

dict_target = {}
for i in range(len(target_word_set)):
    dict_target[target_word_set[i]] = i


def prepare_data(text):
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

def prepare_0(data, word2intdict):
    result_x = []
    result_y = []
    for sent in data:
        length = len(sent)
        bar = [0] * range_len
        boo = []
        for i in range(length):
            boo.append(word2intdict[sent[i]])
        for i in range(length):
            if (sent[i] in target_word_set):
                tmp = boo
                tmp[i] = word2intdict['zzz']
                tmp2 = dict_target[sent[i]]
                bar[tmp2] = 1
                result_x.append(tmp)
                result_y.append(bar)
    return result_x, result_y


def prepare_1(data, word2intdict):
    result_x = []
    result_y = []
    data_int = []
    for i in range(len(data)):
        data_int.append(word2intdict[data[i]])
    sent_len = 3
    for i in range(len(data)):
        if (data[i] in target_word_set):
            bar = [0] * range_len
            bar[dict_target[data[i]]] = 1
            foo = []
            result_y.append(bar)
            for j in range(-1, -1 - sent_len, -1):
                if (i + j < 0):
                    break
                foo.append(data_int[i + j])
            foo.append(word2intdict['zzz'])
            for j in range(1, 1 + sent_len):
                if (i + j == len(data)):
                    break
                foo.append(word2intdict[data[i + j]])
            result_x.append(foo)

    return sent_len * 2 + 1, result_x, result_y

def voc_dict_1000(word_set):

    dict_set = []
    for x in word_set:
        dict_set.append(x)


    fdist = FreqDist(dict_set)
    most_common_word = [w for (w, fq) in fdist.most_common()[:1000]]
    print most_common_word

    #print len(dict_set)
    cnt = 0
    word2intdict = {}
    word2intdict['zzz'] = 0
    for w in most_common_word:
        cnt = cnt + 1
        word2intdict[w] = cnt
    cnt += 1
    for w in dict_set:
        if w not in most_common_word:
            word2intdict[w] = cnt
    word2intdict['though'] = word2intdict['although']
    return word2intdict



##start

cat = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'reviews', 'romance', 'science_fiction']
#cat = ['news']
def data_api(spilt_rate, method = 0):
    raw = brown.tagged_words()
#data = prepare_data(raw)

    tmp = brown.words(categories=cat)
    word2intdict = voc_dict_1000(tmp)
    """
    dict_set = []
    for x in tmp:
        if (x == 'though'):
            dict_set.append('although')
        else:
            dict_set.append(x)
    dict_set.append('zzz')
    dict_set = set(dict_set)
    #print len(dict_set)
    cnt = 0
    word2intdict = {}
    for w in dict_set:
        cnt = cnt + 1
        word2intdict[w] = cnt
    """
    #raw_sent = brown.sents()
    #print('raw_sent: ', len(raw) / len(raw_sent))
    print 'preparing data'
    maxlen = 100
    if (method == 0):
        data_x, data_y = prepare_0(raw_sent[:int(len(raw_sent))], word2intdict)
    elif (method == 1):
        maxlen, data_x, data_y = prepare_1(tmp, word2intdict)
    else:
        print 'you need to choice a preprocessing method!'
    #size = int(spilt_rate * len(data_x))


    train_inds = npr.choice(range(len(data_x)), size = int((1 - spilt_rate) * len(data_x)), replace = False)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(data_x)):
        if i in train_inds:
            X_train.append(data_x[i])
            Y_train.append(data_y[i])
        else:
            X_test.append(data_x[i])
            Y_test.append(data_y[i])

    return maxlen, range_len, (X_train, Y_train), (X_test, Y_test)
    #return maxlen, range_len, (data_x[size:], data_y[size:]), (data_x[:size], data_y[:size])


