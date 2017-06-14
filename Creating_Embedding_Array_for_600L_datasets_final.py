
# coding: utf-8

# In[1]:

import json
from pprint import pprint
from glove import Corpus, Glove
import numpy as np
import hickle as hkl
from gensim.models import KeyedVectors
import sys
import datetime
import re


# In[2]:

print("code running starts at",str(datetime.datetime.now()))


# In[3]:

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


# In[4]:

def add_eos(sent_list):
    new_sent_list = []
    for i in sent_list:
        new_sent_list.append(i+" "+"<eos>")
    return new_sent_list


# In[5]:

contents = add_eos(contents)
titles = add_eos(titles)


# In[6]:

len(titles)


# In[7]:

len(set(titles))


# In[8]:

len(contents)


# In[9]:

len(set(contents))


# In[10]:

def sentences_to_list_of_unique_tokens(sent_list):
    sentences_list_word = []
    i = 0
    for a in sent_list:
        sentences_list_word = sentences_list_word+a
        i+=1
        if i%10000==0:
            print i
    return sentences_list_word

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
        b = list(set(a))
        sentences_list_word.append(b)
        lens.append(len(a))
    return sentences_list_word,lens

def vocab_list(full_list):
    vocab_unique_list = []
    for i in full_list:
        for j in i:
            if j not in vocab_unique_list:
                vocab_unique_list.append(i)
    return vocab_unique_list,len(vocab_unique_list)

def vocab_length(full_list):
    s = 0
    for i in full_list:
        s = s+len(i)
    return s

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


# In[11]:

print("start at",str(datetime.datetime.now()))
sentence_contents,sent_cont_len = sentences_to_list_of_tokens(contents)
print("end at",str(datetime.datetime.now()))

del contents

print("start at",str(datetime.datetime.now()))
sentence_targets,sent_tar_len = sentences_to_list_of_tokens(titles)
print("end at",str(datetime.datetime.now()))

del titles

sentence_contents_new,sent_cont_len_new,sentence_targets_new,sent_tar_len_new = shrink_dataset(sentence_contents,
                                                                                               sentence_targets,
                                                                                               sent_cont_len,
                                                                                               sent_tar_len)


# In[16]:

print("start at",str(datetime.datetime.now()))
sentence_contents_new = [item for sublist in sentence_contents_new for item in sublist]
print("end at",str(datetime.datetime.now()))


# In[17]:

print("start at",str(datetime.datetime.now()))
sentence_targets_new = [item for sublist in sentence_targets_new for item in sublist]
print("end at",str(datetime.datetime.now()))


# In[19]:

sentence_contents_new = list(set(sentence_contents_new))


# In[21]:

sentence_targets_new = list(set(sentence_targets_new))


# In[24]:

corpus_unique_tokens = list(set(sentence_contents_new+sentence_targets_new))


# In[25]:

print("Total number of tokens :",len(corpus_unique_tokens))


# In[26]:

print("glove pretrained reading starts at",str(datetime.datetime.now()))


# In[ ]:

glove_model = KeyedVectors.load_word2vec_format('/mnt/disk3/glove.840B.300d.word2vec.txt')


# In[38]:

print("glove pretrained reading done at",str(datetime.datetime.now()))


# In[2]:

print("for give data trained glove vectors reading starts at",str(datetime.datetime.now()))


# In[40]:

glove_trained_vectors = np.load('/mnt/disk2/sayon/trained_glove_vector_new.npy')


# In[3]:

glove_trained_dictionary = np.load('/mnt/disk2/sayon/trained_glove_vocab_dictionary_new.npy')


# In[4]:

print("for give data trained glove vectors reading done at",str(datetime.datetime.now()))


# In[5]:

glove_trained_dictionary = glove_trained_dictionary.all()


# In[6]:

len(glove_trained_dictionary.keys())


# In[44]:

print("embedding array and dictionary creation starts at",str(datetime.datetime.now()))


# In[67]:

gm_std = glove_model.index2word
gm_trained = glove_trained_dictionary.keys()


# In[ ]:

embedding_array = []
embedding_dict_final = {}
s = 0
x = 0
counter = 0
for j in corpus_unique_tokens:
            if j in gm_std:
                embedding_dict_final[j] = s
                embedding_array.append(glove_model[j])
            elif j in gm_trained:
                embedding_dict_final[j] = s
                embedding_array.append(glove_trained_vectors[glove_trained_dictionary[j]])
            else:
                embedding_dict_final[j] = s
                embedding_array.append(np.random.uniform(high=1,low=-1,size=300))
                x+=1
            s+=1
            counter+=1
            if counter%1000 == 0:
                print counter,"words added","and x words being",x


# In[68]:

print("embedding array and dictionary creation ends at",str(datetime.datetime.now()))


# In[69]:

print("saving of embedding array starts at",str(datetime.datetime.now()))


# In[ ]:

embedding_array = np.asarray(embedding_array)
np.save("/mnt/disk2/sayon/embedding_array_600L_docs.npy",embedding_array)


# In[ ]:

print("saving of embedding array ends at",str(datetime.datetime.now()))


# In[ ]:

print("saving of embedding dictionary starts at",str(datetime.datetime.now()))


# In[ ]:

np.save("/mnt/disk2/sayon/embedding_dict_final_600L_docs.npy",embedding_dict_final)


# In[ ]:

print("saving of embedding dictionary ends at",str(datetime.datetime.now()))


# In[ ]:



