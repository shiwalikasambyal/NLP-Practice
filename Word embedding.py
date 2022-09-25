#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPLEMENTING WORD EMBEDDING


# In[1]:


##tensorflow >2.0
from tensorflow.keras.preprocessing.text import one_hot### Libraries USed Tensorflow> 2.0  and keras


# In[2]:


### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]


# In[3]:


sent


# In[4]:


### Vocabulary size
voc_size=10000


# In[5]:


#one hot representation
onehot_repr=[one_hot(words,voc_size)for words in sent] 
print(onehot_repr)


# In[6]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


# In[7]:


import numpy as np


# In[8]:


sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[9]:


dim=10


# In[10]:


model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')


# In[11]:


model.summary()


# In[12]:


print(model.predict(embedded_docs))


# In[13]:


embedded_docs[0] 
##here only first sentence is taken 
## 


# In[14]:


print(model.predict(embedded_docs)[0])
# each word of the first sentence is converted into 10 dimension that we can count as well
# it is the representation of each word from first sentence


# In[ ]:




