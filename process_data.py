# %%
import os
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm

tqdm.pandas()
import multiprocessing as mp
import re
from multiprocessing import Pool

import matplotlib.pyplot as plt
import nltk
import numpy as np
from tqdm import tqdm

pd.options.mode.chained_assignment = None
from format_dataset import *

country_name = "us"
news_type = 'covid' # (reg or covid)
# news_type = "reg"
date = "april_2020-april_2021"

# Naming convention: Country_NewsType_article-date-range_(todays date)
df_name = f"{country_name}_{news_type}_{date}_({datetime.today().date()}).csv"
print(f"Dataframe Name: {df_name}")

# %%
dataset_name = 'us-no-newsstream-full-timeline'
processed_lists = process_dataset(dataset_name, article_skip=1)

columns=['article_id','date','publisher','title', 'text',
         'language', 'page_num', 'source_type',
        'city', 'author', 'LexileScore']
df = pd.DataFrame(processed_lists, columns=columns)

# %%
before_remove_errors = len(df)
df = df[df.text != 'Error in processing document']
df = df[df.language == 'English'] 
df = df[df.source_type != 'Multimedia']
df['text'] = df['text'].astype('string')
df = df[df.title != 'Cleveland COVID-19 Vaccine Locations']
print(f'{len(df)} Number of articles to analyze. \n {before_remove_errors - len(df)} articles lost.')

# %%
cnn_df = pd.read_csv(f'cnn us_covid_april_2020-april_2021_(2022-09-18).csv')
df = pd.concat([df,cnn_df], axis=0, ignore_index=True)
df.reset_index(drop=True, inplace=True)
df = base_text_formatting(df)
df['text_len'] = df.text.str.split().apply(len)
df = df[(df.text_len > 325) & (df.text_len < 4000)]

sent_tokenizer = nltk.data.load('nltk_tokenizer/punkt/english.pickle')
with mp.Pool(3) as pool:
    df['sentences'] = pool.map(sent_tokenizer.tokenize, df['text'])
print(f'Number of unique articles: {df.article_id.nunique()}')
articles_before = df.article_id.nunique()

def sentences_keep(sentences):
    try:
        if len(sentences) >= 19:
            # first 15 sentences and last 3 sentences not including the
            # last sentence since it is usually an advertisement.
            return sentences[:15] + sentences[-4:-1]
        elif (len(sentences) <= 19) & (len(sentences) >= 5):
            return sentences
        else:
            return np.nan
    except:
        return np.nan

# First we only keep 20 sentences from the article.
df.sentences = df.sentences.apply(lambda x: sentences_keep(x)) 
df.dropna(axis=0, subset=['sentences'], inplace=True)
print(f'Articles Lost from being too short: {articles_before - df.article_id.nunique()}')
df['filtered_text'] = df['sentences'].str.join(' ')
# Then we count the number of keywords in the text.
df = experimental_count_keywords(df)
df = experimental_covid_article_filtering(df)
print(f'Number of unique articles: {df.article_id.nunique()}')

print(df.groupby('source_type').count()['article_id'] / df.groupby('source_type').count()['article_id'].sum())

def remove_non_relevant_content(sentences):
    copy_sentences = sentences.copy()
    check_one = 'Newstex Authoritative Content is not'
    check_two = 'The material and information provided in Newstex'
    check_three = 'Sign up for our'
    check_four = 'Neither newstex nor its re-distributors'
    check_five = 'Please wait for the page to reload'
    for sentence_num, sentence in enumerate(copy_sentences):
        if (bool(re.search(check_one, sentence, re.I)) or \
        bool(re.search(check_two, sentence, re.I)) or \
        bool(re.search(check_three, sentence, re.I)) or \
        bool(re.search(check_four, sentence, re.I)) or \
        bool(re.search(check_five, sentence, re.I))) and (sentence_num >= 9):
            
            print(f'we got one {sentence_num}')
            return sentences[:sentence_num]
    return sentences
    
df['sentences'] = df['sentences'].apply(remove_non_relevant_content) 

def keep_pairs(lst):
    """ Make sentences into groups of three.
    If the article is full length, this will lead to 6 predictions per article.
    Grouping makes predictions faster and more accurate. 
    However, groups larger than 3 will usually go above Roberta's character limit.
    """
    return [' '.join(x) for x in zip(lst[0::3], lst[1::3], lst[2::3])]
df['pairs'] = df.sentences.apply(keep_pairs)

# %%
df.drop(['text', 'language', 'sentences', 'filtered_text'],axis=1,inplace=True)
pre_explode = df.drop('pairs', axis=1) # we only use pre_explode as an article_id reference.
pre_explode.to_csv('csv/pre_explode_' + df_name) 

df = df.explode('pairs') #This keeps it in the format required for data loader.
print(f'Dataframe {df_name} goes from {df.date.min()} to {df.date.max()}.')
print(f'Dataframe {df_name} has {df.article_id.nunique()} unique articles.')
df.page_num.fillna('None', inplace=True)

df.to_csv('csv/no_txt_' + df_name)
print(f"Finished {df_name}")


