import os
from lxml import etree
from bs4 import BeautifulSoup
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import re
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment = None


def getxmlcontent(root):
    """ Helper function for make_lists
    """
    if root.find('.//HiddenText') is not None:
        return(root.find('.//HiddenText').text)
    elif root.find('.//Text') is not None:
        return(root.find('.//Text').text)   
    else:
        return None
    
# Extract the necessary goid, text, and date content from the XML files
# Set up for multiprocessing--for a single file
def make_lists(article, dataset_name):
    """ Needs to be declared outside of process_dataset() because of multiprocessing pickling.
    """
    dataset_prefix = '/home/ec2-user/SageMaker/data/'
    try: 
        tree = etree.parse(dataset_prefix + dataset_name + '/' + article)
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

def process_dataset(dataset_name):
    """ Processes ProQuest XML files in parallel.

    Args:
        dataset_name (String): Name of ProQuest database saved in the "data" folder.

    Returns: A tuple with the specified return data in make_lists().
    """
    dataset_prefix = '/home/ec2-user/SageMaker/data/'
    articles = os.listdir(dataset_prefix + dataset_name + '/')

    # Define a function to get the text content that is needed from the XML articles available in the dataset

    # Check core count
    num_cores = mp.cpu_count()
    print(f"{num_cores} CPU cores available.")

    # When using multiple processes, important to eventually close them to avoid memory/resource leaks
    try:
        p = Pool(processes=num_cores - 1)
        processed_lists = p.map(make_lists, args=(articles[:],dataset_name))
    except:
        print("Error in processing document")
    finally:
        p.close()

    return processed_lists

def base_text_formatting(dataframe):
    dataframe.text = dataframe.text.str.replace('\xa0', ' ')
    dataframe.text = dataframe.text.str.strip()
    dataframe.text = dataframe.text.str.replace('\n',' ')
    dataframe.text = dataframe.text.str.replace('(?<=\.)(?=[A-Z]\B)', ' ', regex=True)
    dataframe.text = dataframe.text.str.replace('sars.cov.2', 'covid', flags=re.IGNORECASE, regex=True)
    dataframe.text = dataframe.text.str.replace('coronavirus', 'covid', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('covid.19', 'covid', flags=re.IGNORECASE, regex=True)
    return dataframe


def count_keywords(dataframe, regular_news=False):
    covid_keywords = ('covid','corona', 'virus', 'variant', 'vaccine',
                        'hospital', 'cdc','icu', 'lockdown', 'omicron', 'delta', 
                        'ventilator','infect','mask', 'cases', 'n95')
    initial_series = dataframe.text.str.count(covid_keywords[0], flags=re.IGNORECASE)
    for keyword in covid_keywords[1:]:
        temp_series = dataframe.text.str.count(keyword, flags=re.IGNORECASE)
        initial_series += temp_series
    dataframe['keyword_len'] = initial_series

    if not regular_news:
        dataframe = dataframe[(dataframe.text.str.count('covid', flags=re.IGNORECASE) >= 2) | 
                            (dataframe.text.str.count('omicron', flags=re.IGNORECASE) >= 2) | 
                            (dataframe.text.str.count('delta', flags=re.IGNORECASE) >= 2)]

    dataframe['text_len'] = dataframe.text.str.split().apply(len)
    dataframe = dataframe[(dataframe.text_len > 325) & (dataframe.text_len < 3500)]
    return dataframe

def keep_articles_based_on_keywords(dataframe):
    dataframe = dataframe[(((dataframe.keyword_len >= 5) & (dataframe.keyword_len < 40)) & (dataframe.text_len < 500))|
                    ((dataframe.keyword_len >= 7)  & (dataframe.text_len < 750))|
                    (((dataframe.keyword_len >= 10) & (dataframe.text_len <= 1050)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 14) & (dataframe.text_len <= 1500)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 20) & (dataframe.text_len <= 2000)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 30) & (dataframe.text_len <= 3200)) & (dataframe.text_len > 750))]
    return dataframe

def covid_format_text(dataframe):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe)

    dataframe.text = dataframe.text.str.replace('we\'ll','we will', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('didn\'t','did not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('doesn\'t','does not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('what\'s','what is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('haven\'t','have not', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('that\'s','that is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('he\'s','he is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('she\'s','she is', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('don\'t','do not', flags=re.IGNORECASE)
    dataframe = keep_articles_based_on_keywords(dataframe)
    return dataframe 

def headline_formatting(dataframe):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe)
    dataframe = keep_articles_based_on_keywords(dataframe)
    dataframe = dataframe[['article_id', 'date', 'publisher', 'title', 'page_num']]
    return dataframe
    
def regular_news_formatting(dataframe, num_articles_to_sample=150000):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe, regular_news=True)
    dataframe = dataframe[(dataframe.text.str.count('covid', flags=re.IGNORECASE) <= 1)]
    dataframe = dataframe[dataframe.keyword_len <= 3]
    dataframe = dataframe.sample(num_articles_to_sample)
    return dataframe
