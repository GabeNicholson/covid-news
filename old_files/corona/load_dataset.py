import os
import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup
from tqdm import tqdm
tqdm.pandas()
import multiprocessing as mp
from multiprocessing import Pool
import re
import numpy as np
from tqdm import tqdm
import nltk
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

def process_dataset(dataset_name):
    dataset_prefix = '/home/ec2-user/SageMaker/data/'
    articles = os.listdir(dataset_prefix + dataset_name + '/')
    corpus_directory = dataset_prefix

    # Define a function to get the text content that is needed from the XML articles available in the dataset
    def getxmlcontent(root):
        if root.find('.//HiddenText') is not None:
            return(root.find('.//HiddenText').text)
        elif root.find('.//Text') is not None:
            return(root.find('.//Text').text)   
        else:
            return None
        
    # Extract the necessary goid, text, and date content from the XML files
    # Set up for multiprocessing--for a single file
    def make_lists(article):
        try: 
            tree = etree.parse(corpus_directory + dataset_name + '/' + article)
            root = tree.getroot()
            if getxmlcontent(root):
                soup = BeautifulSoup(getxmlcontent(root))
                text = soup.get_text().replace('\\n','\n')
            else:
                text = 'Error in processing document'
            date = root.find('.//NumericDate').text
            publication = root.find('.//SortTitle').text
            title = root.find('.//Title').text
            language = root.find('.//RawLang').text
            source_type = root.find('.//SourceRollupType').text
            try:
                page_num = root.find('.//StartPage').text
            except:
                page_num = 'None'
                
        except AttributeError:
            # Error logging - will show filename if there is a problem processing it
            print("Attribute Error" + article)
        return article, date, publication, title, text , language, page_num, source_type


    # Check core count
    num_cores = mp.cpu_count()
    print(num_cores)

    # When using multiple processes, important to eventually close them to avoid memory/resource leaks
    try:
        # Define a thread Pool to process multiple XML files simultaneously
        # Default set to num_cores - 1, but may change number of processes depending on instance
        p = Pool(processes=4)
        
        # Apply function with Pool to corpus
        processed_lists = p.map(make_lists, articles[:])
    except:
        print("Error in processing document")
    finally:
        p.close()

    return processed_lists

def base_text_format(dataframe):
    dataframe.text = dataframe.text.str.replace('\xa0', ' ')
    dataframe.text = dataframe.text.str.strip()
    dataframe.text = dataframe.text.str.replace('\n',' ')
    dataframe.text = dataframe.text.str.replace('(?<=\.)(?=[A-Z]\B)', ' ', regex=True)
    dataframe.text = dataframe.text.str.replace('sars.cov.2', 'covid', flags=re.IGNORECASE, regex=True)
    dataframe.text = dataframe.text.str.replace('coronavirus', 'covid', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('covid.19', 'covid', flags=re.IGNORECASE, regex=True)
    return dataframe
    
def format_text(dataframe):
    dataframe = base_text_format(dataframe)

    covid_keywords = ('covid','corona', 'virus', 'variant', 'vaccine',
                        'hospital', 'cdc','icu', 'lockdown', 'omicron', 'delta', 
                        'ventilator','infect','mask', 'cases', 'n95')

    initial_series = dataframe.text.str.count(covid_keywords[0], flags=re.IGNORECASE)
    for keyword in covid_keywords[1:]:
        temp_series = dataframe.text.str.count(keyword, flags=re.IGNORECASE)
        initial_series += temp_series
    dataframe['keyword_len'] = initial_series

    dataframe = dataframe[(dataframe.text.str.count('covid', flags=re.IGNORECASE) >= 2)   | 
            (dataframe.text.str.count('omicron', flags=re.IGNORECASE) >= 2) | 
            (dataframe.text.str.count('delta', flags=re.IGNORECASE) >= 2)]

    dataframe['text_len'] = dataframe.text.str.split().apply(len)
    dataframe = dataframe[dataframe.text_len > 325]
    dataframe = dataframe[dataframe.text_len < 3500]

    dataframe.text = dataframe.text.str.replace('we\'ll','we will', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('didn\'t','did not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('doesn\'t','does not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('what\'s','what is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('haven\'t','have not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('that\'s','that is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('he\'s','he is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('she\'s','she is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('don\'t','do not', flags=re.IGNORECASE)


    dataframe = dataframe[(((dataframe.keyword_len >= 5) & (dataframe.keyword_len < 40)) & (dataframe.text_len < 500))|
                    ((dataframe.keyword_len >= 7)  & (dataframe.text_len < 750))|
                    (((dataframe.keyword_len >= 10) & (dataframe.text_len <= 1050)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 14) & (dataframe.text_len <= 1500)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 20) & (dataframe.text_len <= 2000)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 30) & (dataframe.text_len <= 3200)) & (dataframe.text_len > 750))]



def headline_formatting(dataset):
    pass



def regular_news_formatting(dataset):
    df.text = df.text.str.replace('\xa0', ' ')
    df.text = df.text.str.strip()
    df.text = df.text.str.replace('\n',' ')
    df.text = df.text.str.replace('(?<=\.)(?=[A-Z]\B)', ' ', regex=True)
    df.text = df.text.str.replace('sars.cov.2', 'covid', flags=re.IGNORECASE, regex=True)
    df.text = df.text.str.replace('coronavirus', 'covid', flags=re.IGNORECASE)
    df.text = df.text.str.replace('covid.19', 'covid', flags=re.IGNORECASE, regex=True)



# if __name__ == '__main__':
#     create_dataset(dataset_name)