#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import zipfile
import re
import os


# In[70]:


def loadcorpus(targetdir,endpoint=False):
    break_flag = False
    texts = pd.DataFrame()
    for file1 in os.listdir(targetdir):
        file1name = os.fsdecode(file1)
        if file1name.startswith('text'):
            zfile = zipfile.ZipFile(targetdir + '/' + file1)
            for file2 in zfile.namelist():
                file2name = os.fsdecode(file2)
                #optional endpoint if you only want a portion of the NOW corpus
                if file2name == endpoint:
                    break_flag = True
                    break
                print(file2name)
                data = pd.read_fwf(zfile.open(f'{file2name}'),colspecs=[(2,10),(11,None)],encoding='utf-8',names=['id','body'])
                texts = texts.append(data,ignore_index=True)
        if break_flag == True:
            break
    return texts


# In[71]:


texts = loadcorpus('/Users/gabrielnicholson/Desktop/corona/corona-21-05', '21-05-31.txt')


# In[72]:


def loadmetadata(targetdir):
    metadata = pd.DataFrame()
    for file1 in os.listdir(targetdir):
        file1name = os.fsdecode(file1)
        print(file1name)
        if file1name.startswith('now_sources') or file1name.startswith('sources'):
            zfile = zipfile.ZipFile(targetdir + '/' + file1)
            for file2 in zfile.namelist():
                file2name = os.fsdecode(file2)
                print(file2name)
                data = pd.read_csv(zfile.open(f'{file2name}'),sep='\t',error_bad_lines=False,engine='python',encoding='latin1',names=['id','length','date','country','publisher','url','snippet'])
                metadata = metadata.append(data,ignore_index=True)
    return metadata


# In[73]:


metadata = loadmetadata('/Users/gabrielnicholson/Desktop/corona/corona-21-05')


# In[74]:


metadata


# In[75]:


texts = texts[1:]


# In[77]:


def remove_weird(x):
    try:
        return int(x)
    except:
        return 'drop'
        


# In[78]:


texts = texts[texts['id'].apply(lambda x: remove_weird(x)) != 'drop']


# In[79]:


texts['id'] = texts['id'].apply(lambda x: int(x))


# In[80]:


merged = pd.merge(metadata,texts,on='id',how='inner')


# In[82]:


def reformat(x):
    return re.sub(r'(<p>|@ )','', x)


# In[83]:


merged['body'] = merged['body'].apply(lambda x: str(x))
merged['body'] = merged['body'].apply(reformat)


# In[84]:


merged.to_csv('corona-21-05.csv')


# In[3]:


df = pd.read_csv('corona-21-05.csv')


# In[17]:


print(df['snippet'][1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




