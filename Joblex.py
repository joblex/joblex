#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
from gensim.parsing.preprocessing import *
import numpy as np
from sklearn.feature_extraction.text import *


# In[2]:


tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec('glove.twitter.27B.200d.txt', tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)


# In[3]:


class Vec2:
    def __init__(self, gensim_model):
        self.model = gensim_model
        
    def transform(self, X):
        v = []
        for word in X:
            if self.model.__contains__(word): 
                v.append(self.model.get_vector(word))
        if len(v)==0:
            return np.zeros(200)
        return np.mean(v, axis=0)
model2 = Vec2(model)


# In[4]:


def text_preprocess(s):
    s = " ".join(s.split("\n"))
    s = " ".join(s.split(" "))
    s = s.lower().encode('ascii', 'ignore')
    s = re.sub(r'(\bwww\..+?(\s|$))', ' ', s)
    s = re.sub(r'(\bhttps{0,1}\:\/\/.+?(\s|$))', '', s)
    s = re.sub(r'(@.+?(\b|\s|$))', '', s)
    s = re.sub(r'\bRT\b', '', s)
    s = re.sub(r'\s+',' ',s)
    s = strip_tags(s)
    s = strip_punctuation(s)
    s = strip_multiple_whitespaces(s)
    s = strip_numeric(s)
    s = remove_stopwords(s)
    s = s.lower().encode('ascii', 'ignore')
    return s


# In[5]:


ontology_map = {
    "Abilities":['1.A.1.a.1', '1.A.1.a.2', '1.A.1.a.3', '1.A.1.a.4', '1.A.1.b.1',
       '1.A.1.b.2', '1.A.1.b.3', '1.A.1.b.4', '1.A.1.b.5', '1.A.1.b.6',
       '1.A.1.b.7', '1.A.1.c.1', '1.A.1.c.2', '1.A.1.d.1', '1.A.1.e.1',
       '1.A.1.e.2', '1.A.1.e.3', '1.A.1.f.1', '1.A.1.f.2', '1.A.1.g.1',
       '1.A.1.g.2', '1.A.2.a.1', '1.A.2.a.2', '1.A.2.a.3', '1.A.2.b.1',
       '1.A.2.b.2', '1.A.2.b.3', '1.A.2.b.4', '1.A.2.c.1', '1.A.2.c.2',
       '1.A.2.c.3', '1.A.3.a.1', '1.A.3.a.2', '1.A.3.a.3', '1.A.3.a.4',
       '1.A.3.b.1', '1.A.3.c.1', '1.A.3.c.2', '1.A.3.c.3', '1.A.3.c.4',
       '1.A.4.a.1', '1.A.4.a.2', '1.A.4.a.3', '1.A.4.a.4', '1.A.4.a.5',
       '1.A.4.a.6', '1.A.4.a.7', '1.A.4.b.1', '1.A.4.b.2', '1.A.4.b.3',
       '1.A.4.b.4', '1.A.4.b.5'],
    "Interests":["1.B.1.a", "1.B.1.b", "1.B.1.c", "1.B.1.d", "1.B.1.e", "1.B.1.f"],
    "Knowledge":['2.C.1.a', '2.C.1.b', '2.C.1.c', '2.C.1.d', '2.C.1.e', '2.C.1.f',
       '2.C.2.a', '2.C.2.b', '2.C.3.a', '2.C.3.b', '2.C.3.c', '2.C.3.d',
       '2.C.3.e', '2.C.4.a', '2.C.4.b', '2.C.4.c', '2.C.4.d', '2.C.4.e',
       '2.C.4.f', '2.C.4.g', '2.C.5.a', '2.C.5.b', '2.C.7.a', '2.C.7.b',
       '2.C.7.c', '2.C.7.d', '2.C.7.e', '2.C.8.a', '2.C.8.b', '2.C.9.a',
       '2.C.9.b', '2.C.10', '2.C.6'],
    "Skills":['2.A.1.a', '2.A.1.b', '2.A.1.c', '2.A.1.d', '2.A.1.e', '2.A.1.f',
       '2.A.2.a', '2.A.2.b', '2.A.2.c', '2.A.2.d', '2.B.1.a', '2.B.1.b',
       '2.B.1.c', '2.B.1.d', '2.B.1.e', '2.B.1.f', '2.B.2.i', '2.B.3.a',
       '2.B.3.b', '2.B.3.c', '2.B.3.d', '2.B.3.e', '2.B.3.g', '2.B.3.h',
       '2.B.3.j', '2.B.3.k', '2.B.3.l', '2.B.3.m', '2.B.4.e', '2.B.4.g',
       '2.B.4.h', '2.B.5.a', '2.B.5.b', '2.B.5.c', '2.B.5.d'],
    "Work Activities":['4.A.1.a.1', '4.A.1.a.2', '4.A.1.b.1', '4.A.1.b.2', '4.A.1.b.3',
       '4.A.2.a.1', '4.A.2.a.2', '4.A.2.a.3', '4.A.2.a.4', '4.A.2.b.1',
       '4.A.2.b.2', '4.A.2.b.3', '4.A.2.b.4', '4.A.2.b.5', '4.A.2.b.6',
       '4.A.3.a.1', '4.A.3.a.2', '4.A.3.a.3', '4.A.3.a.4', '4.A.3.b.1',
       '4.A.3.b.2', '4.A.3.b.4', '4.A.3.b.5', '4.A.3.b.6', '4.A.4.a.1',
       '4.A.4.a.2', '4.A.4.a.3', '4.A.4.a.4', '4.A.4.a.5', '4.A.4.a.6',
       '4.A.4.a.7', '4.A.4.a.8', '4.A.4.b.1', '4.A.4.b.2', '4.A.4.b.3',
       '4.A.4.b.4', '4.A.4.b.5', '4.A.4.b.6', '4.A.4.c.1', '4.A.4.c.2',
       '4.A.4.c.3'],
    "Work Context": ['4.C.1.a.2.c', '4.C.1.a.2.f', '4.C.1.a.2.h',
       '4.C.1.a.2.j', '4.C.1.a.2.l', '4.C.1.a.4',
       '4.C.1.b.1.e', '4.C.1.b.1.f', '4.C.1.b.1.g', '4.C.1.c.1',
       '4.C.1.c.2', '4.C.1.d.1', '4.C.1.d.2', '4.C.1.d.3', 
       '4.C.2.a.1.a', '4.C.2.a.1.b', '4.C.2.a.1.c', '4.C.2.a.1.d',
       '4.C.2.a.1.e', '4.C.2.a.1.f', '4.C.2.a.3',
       '4.C.2.b.1.a', '4.C.2.b.1.b', '4.C.2.b.1.c', '4.C.2.b.1.d',
       '4.C.2.b.1.e', '4.C.2.b.1.f', '4.C.2.c.1.a',
       '4.C.2.c.1.b', '4.C.2.c.1.c', '4.C.2.c.1.d', '4.C.2.c.1.e',
       '4.C.2.c.1.f', '4.C.2.c.2', '4.C.2.c.3', 
       '4.C.2.d.1.a', '4.C.2.d.1.b', '4.C.2.d.1.c', '4.C.2.d.1.d',
       '4.C.2.d.1.e', '4.C.2.d.1.f', '4.C.2.d.1.g', '4.C.2.d.1.h',
       '4.C.2.d.1.i',  '4.C.2.e.1.d', '4.C.2.e.1.e',
       '4.C.3.a.1', '4.C.3.a.2.a', '4.C.3.a.2.b',
       '4.C.3.a.4', '4.C.3.b.2', '4.C.3.b.4', '4.C.3.b.7', '4.C.3.b.8',
       '4.C.3.c.1', '4.C.3.d.1', '4.C.3.d.3', '4.C.3.d.4', '4.C.3.d.8'],
    "Work Styles":['1.C.1.a', '1.C.1.b', '1.C.1.c', '1.C.2.b', '1.C.3.a', '1.C.3.b',
       '1.C.3.c', '1.C.4.a', '1.C.4.b', '1.C.4.c', '1.C.5.a', '1.C.5.b',
       '1.C.5.c', '1.C.7.a', '1.C.7.b','1.C.6'],
    "Work Values":[
        '1.B.2.a', '1.B.2.b', '1.B.2.c', '1.B.2.d', '1.B.2.e', '1.B.2.f'
    ]
    
}

ontology_df=pd.read_csv("Content Model Reference.txt",sep="\t")


# In[6]:


onet_ontology = {}

for key in ontology_map.keys():
    category = key
    onet_ontology[key] = {}
    sub_df = ontology_df[ontology_df["Element ID"].isin(ontology_map[key])]
    for idx, row in sub_df.iterrows():
        onet_ontology[key][row['Element Name']] = row['Description']


# In[8]:


CAT = []
SUBCAT = []
TEXT = []
SIMS = {}
dim = 30
for i in range(dim):
    SIMS[i] = []
for cat in onet_ontology:
    for subcat in onet_ontology[cat]:
        s = onet_ontology[cat][subcat]
        s1 = text_preprocess(s)
        t = s1.split(' ')[0]
        vec = model2.transform(s1.split(' ')).flatten()
        CAT.append(cat)
        SUBCAT.append(subcat)
        TEXT.append(s)
        sims = model.similar_by_vector(vec, topn=5000)
        t = 0
        for i in range(5000):
            word = sims[i][0]
            word = strip_tags(word)
            word = strip_punctuation(word)
            word = remove_stopwords(word)
            if word == "b''" or word=="n't" or len(word)<5 or sims[i][1]>1:
                continue
            if word in ['thing', 'things', 'great', 'good', 'best', 'going', 'knowledge']:
                continue
            if word != "":
                SIMS[t].append("{}, {}".format(sims[i][0], round(sims[i][1],2)))
                t+=1
                if t==dim:
                    break

Similar_Sentences = pd.DataFrame()
Similar_Sentences['Category'] = CAT
Similar_Sentences['Sub Category'] = SUBCAT
Similar_Sentences['Text'] = TEXT
for i in range(dim):
    Similar_Sentences[i] = SIMS[i]
Similar_Sentences


# In[10]:


Similar_Sentences.to_csv("joblex.tsv", sep="\t", index=False)


# In[ ]:




