
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
from matplotlib import pyplot as plt
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


# In[6]:

print("start at",str(datetime.datetime.now()))
sentence_contents,sent_cont_len = sentences_to_list_of_tokens(contents)
print("end at",str(datetime.datetime.now()))


# In[7]:

print("start at",str(datetime.datetime.now()))
sentence_targets,sent_tar_len = sentences_to_list_of_tokens(titles)
print("end at",str(datetime.datetime.now()))


# In[8]:

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


# In[9]:

sentence_contents_new,sent_cont_len_new,sentence_targets_new,sent_tar_len_new = shrink_dataset(sentence_contents,
                                                                                               sentence_targets,
                                                                                               sent_cont_len,
                                                                                               sent_tar_len)


# In[9]:

embedding_array = np.load("/mnt/disk2/sayon/embedding_array_600L_docs.npy")
embedding_dict_final = np.load("/mnt/disk2/sayon/embedding_dict_final_600L_docs.npy")


# In[10]:

embedding_dict_final = embedding_dict_final.all()


# In[10]:

max_length_contents = max([len(i) for i in sentence_contents_new])
max_length_targets = max([len(i) for i in sentence_targets_new])
min_length_contents = min([len(i) for i in sentence_contents_new])
min_length_targets = min([len(i) for i in sentence_targets_new])


# In[11]:

max_length_contents


# In[12]:

max_length_targets


# In[13]:

min_length_contents


# In[14]:

min_length_targets


# In[15]:
'''
data=[len(i) for i in sentence_contents_new]
plt.figure(figsize=(10,5))
plt.hist(data,bins=5)
plt.show()


# In[16]:

plt.figure(figsize=(15,10))
plt.plot(data)
plt.show()


# In[17]:

#capping max length for sentence_contents to 2100


# In[18]:

data=[len(i) for i in sentence_targets_new]
plt.figure(figsize=(10,5))
plt.hist(data,bins=5)
plt.show()


# In[19]:

plt.figure(figsize=(15,10))
plt.plot(data)
plt.show()

'''
# In[20]:

cap_length_encoder = 600
cap_length_decoder = 56


# In[21]:

sentence_contents = sentence_contents_new
sentence_targets = sentence_targets_new


# In[22]:

#resizing contents and targets
def resizing(lsent,cap_len):
    sent_list_new = []
    for i in lsent:
        sent_list_new.append(i[:cap_len])
    return sent_list_new


# In[23]:

sentence_contents_resized = resizing(sentence_contents,cap_length_encoder)
sentence_targets_resized = resizing(sentence_targets,cap_length_decoder)


# In[24]:
'''
data=[len(i) for i in sentence_contents_resized]
plt.figure(figsize=(10,5))
plt.hist(data,bins=5)
plt.show()


# In[25]:

plt.figure(figsize=(15,10))
plt.plot(data)
plt.show()


# In[32]:

data=[len(i) for i in sentence_targets_resized]
plt.figure(figsize=(10,5))
plt.hist(data,bins=5)
plt.show()


# In[33]:

plt.figure(figsize=(15,10))
plt.plot(data)
plt.show()

'''
# In[34]:

#converting_data_to_embedding_indices
def reframed_doc_to_embeddings(lsent,embedding_dict_final):
    sent_reframed = []
    for i in lsent:
        sent = []
        for j in i:
            #print embedding_dict_final[j]
            sent.append(embedding_dict_final[j])
        sent_reframed.append(sent)
    return np.array(sent_reframed)


# In[35]:

sentence_contents_resized_array = reframed_doc_to_embeddings(sentence_contents_resized,embedding_dict_final)
sentence_targets_resized_array = reframed_doc_to_embeddings(sentence_targets_resized,embedding_dict_final)


# In[36]:

#var_length_calc_before_padding
def var_length(lsent):
    return [len(i) for i in lsent]


# In[37]:

var_lengths_contents = var_length(sentence_contents_resized)
var_lengths_targets = var_length(sentence_targets_resized)


# In[38]:

#padding 
def padding(lsent,cap_length):
    sent_padded = []
    for i in lsent:
        num = cap_length - len(i)
        new_i = list(i) + list(np.zeros(num,dtype=int))
        sent_padded.append(new_i)
    return np.array(sent_padded)


# In[39]:

sentence_contents_resized_array_padded = padding(sentence_contents_resized_array,cap_length_encoder)
sentence_targets_resized_array_padded = padding(sentence_targets_resized_array,cap_length_decoder)


# In[40]:

print sentence_contents_resized_array_padded.shape


# In[41]:

print sentence_targets_resized_array_padded.shape


# In[42]:

np.save("/mnt/disk2/sayon/contents_data_embedded_format_from_600L_docs.npy",sentence_contents_resized_array_padded)
np.save("/mnt/disk2/sayon/targets_data_embedded_format_from_600L_docs.npy",sentence_targets_resized_array_padded)
np.save("/mnt/disk2/sayon/contents_var_length_format_from_600L_docs.npy",var_lengths_contents)
np.save("/mnt/disk2/sayon/targets_var_length_format_from_600L_docs.npy",var_lengths_targets)



np.save("/mnt/disk2/sayon/targets_data_embedded_not_padded_format_from_600L_docs.npy",sentence_targets_resized_array)



