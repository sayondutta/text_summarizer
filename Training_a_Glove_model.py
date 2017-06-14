
# coding: utf-8

# In[14]:

import json
from pprint import pprint
from glove import Corpus, Glove
import numpy as np
import hickle as hkl
from gensim.models import KeyedVectors
import datetime
import re


# In[2]:

contents = []
titles = []
i = 0
with open('/mnt/disk3/signalmedia-1m.jsonl') as data_file:    
    for line in data_file:
        dictline = json.loads(line)
        contents.append(dictline['content'].encode('ascii', 'ignore'))
        titles.append(dictline['title'].encode('ascii', 'ignore'))
        i+=1
        if i%100000==0:
            print i


# In[3]:

def add_eos(sent_list):
    new_sent_list = []
    for i in sent_list:
        new_sent_list.append(i+" "+"<eos>")
    return new_sent_list


# In[4]:

contents = add_eos(contents)
titles = add_eos(titles)


# In[5]:

len(titles)


# In[6]:

len(set(titles))


# In[7]:

len(contents)


# In[8]:

len(set(contents))


# In[9]:

#import itertools
#from gensim.models.word2vec import Text8Corpus


# In[10]:

#sentences = list(itertools.islice(Text8Corpus('text8'),None))


# In[11]:

#vocabulary list containing unique tokens, number of unique tokens total, list containing sentence as list of tokens


# In[12]:

def sentences_to_list_of_tokens(sent_list):
    sentences_list_word = []
    lens = []
    for a in sent_list:
        a = re.sub('\([^)]*\)', '',a)
        a = a.replace('\n',' ')
        a = a.replace('\u',' ')
        a = a.replace('\t',' ')
        a = a.replace('\r',' ')
        a = a.replace(',',' ')
        a = a.replace(';',' ')
        a = a.replace('(',' ')
        a = a.replace(')',' ')
        a = a.replace('.',' ')
        a = a.replace('\\',' ')
        a = a.replace('/',' ')
        a = a.replace('\"',' ')
        a = a.replace('-',' ')
        a = a.split()
        sentences_list_word.append(a)
        lens.append(len(a))
    return sentences_list_word,lens

def vocab_list(full_list):
    vocab_unique_list = []
    for i in full_list:
        for j in i:
            if j not in vocab_unique_list:
                vocab_unique_list.append(i)
    return vocab_unique_list,len(vocab_unique_list)

def shrink_dataset(sc,st,scl,stl):
    scn = []
    cln = []
    stn = []
    tln = []
    for i in range(len(scl)):
        if scl[i]>449 and scl[i]<601:
            scn.append(sc[i])
            stn.append(st[i])
            cln.append(scl[i])
            tln.append(stl[i])
    return scn,cln,stn,tln


# In[15]:

print("start at",str(datetime.datetime.now()))
sentence_contents,sent_cont_len = sentences_to_list_of_tokens(contents)
print("end at",str(datetime.datetime.now()))


# In[16]:

print("start at",str(datetime.datetime.now()))
sentence_targets,sent_tar_len = sentences_to_list_of_tokens(titles)
print("end at",str(datetime.datetime.now()))


# In[17]:

sentence_contents_new,sent_cont_len_new,sentence_targets_new,sent_tar_len_new = shrink_dataset(sentence_contents,
                                                                                               sentence_targets,
                                                                                               sent_cont_len,
                                                                                               sent_tar_len)


# In[18]:

sentence_corpus = sentence_contents_new+sentence_targets_new


# In[19]:

corpus = Corpus()


# In[20]:

corpus.fit(sentence_corpus,window=10)


# In[21]:

import sys


# In[23]:

glove = Glove(no_components=300, learning_rate=0.01)


# In[24]:

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)


# In[25]:

glove.add_dictionary(corpus.dictionary)


# In[26]:

len(glove.dictionary)


# In[29]:

np.save("/mnt/disk2/sayon/trained_glove_vocab_dictionary_new.npy",glove.dictionary)


# In[ ]:

np.save("/mnt/disk2/sayon/trained_glove_vector_new.npy",glove.word_vectors)


# In[ ]:

#After this open 


# In[32]:

#glove.load('/mnt/disk2/gloVe.model')
#glove_model = KeyedVectors.load_word2vec_format('/mnt/disk2/glove.840B.300d.word2vec.txt')


# In[29]:

#glove : new glove vectors trained from the given text
#glove_model : pretrained Glove vectors


# In[27]:

#glove.dictionary


# In[28]:

#glove.word_vectors[glove.dictionary['Eury']]

