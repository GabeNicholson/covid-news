import os
from lxml import etree
from functools import partial
from bs4 import BeautifulSoup
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import re
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment = None

def getxmlcontent(root):
    if root.find('.//HiddenText') is not None:
        return(root.find('.//HiddenText').text)
    elif root.find('.//Text') is not None:
        return(root.find('.//Text').text)   
    else:
        return None
    
# Extract the necessary goid, text, and date content from the XML files
# Set up for multiprocessing--for a single file
def make_lists(article, dataset_name):
    dataset_prefix = '/home/ec2-user/SageMaker/data/'
    try: 
        tree = etree.parse(dataset_prefix + dataset_name + '/' + article)
        root = tree.getroot()
        if getxmlcontent(root):
            soup = BeautifulSoup(getxmlcontent(root), features="lxml")
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

def process_dataset(dataset_name, article_skip=1):
    dataset_prefix = '/home/ec2-user/SageMaker/data/'
    articles = os.listdir(dataset_prefix + dataset_name + '/')

    num_cores = mp.cpu_count()
    print(f"Cores Available: {num_cores}")
    # Get the text content that is needed from the XML articles available in the dataset
    try:
        p = Pool(processes=num_cores - 1)
        prod_x = partial(make_lists, dataset_name=dataset_name)
        processed_lists = p.map(prod_x, articles[::article_skip])
        return processed_lists 
    except:
        print("Error in processing document")
    finally:
        p.close()

def base_text_formatting(dataframe):
    dataframe.text = dataframe.text.str.replace('\xa0', ' ')
    dataframe.text = dataframe.text.str.strip()
    dataframe.text = dataframe.text.str.replace('\n',' ')
    dataframe.text = dataframe.text.str.replace('(?<=\.)(?=[A-Z]\B)', ' ', regex=True)
    dataframe.text = dataframe.text.str.replace('covid', 'covid', flags=re.IGNORECASE, regex=True)
    dataframe.text = dataframe.text.str.replace('sars.cov.2', 'covid', flags=re.IGNORECASE, regex=True)
    dataframe.text = dataframe.text.str.replace('coronavirus', 'covid', flags=re.IGNORECASE)
    dataframe.text = dataframe.text.str.replace('covid.19', 'covid', flags=re.IGNORECASE, regex=True)
    return dataframe

def count_keywords(dataframe, regular_news=False):
    """Counts the number of keywords in each article. This is used to filter
    relevant covid articles to non-relevant covid articles. If it is being used on
    non-covid related news, then the keywords are used to filter out Covid articles.

    Args:
        dataframe (Pandas Dataframe): 
        regular_news (bool, optional): If doing it on regular news article set to True. Otherwise defaults to False.

    Returns:
        Dataframe: dataframe with the text length and number of keywords attached.
    """
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

def experimental_count_keywords(dataframe, regular_news=False):
    """Counts the number of keywords in each article. This is used to filter
    relevant covid articles to non-relevant covid articles. If it is being used on
    non-covid related news, then the keywords are used to filter out Covid articles.

    Args:
        dataframe (Pandas Dataframe): 
        regular_news (bool, optional): If doing it on regular news article set to True. Otherwise defaults to False.

    Returns:
        Dataframe: dataframe with the text length and number of keywords attached.
    """
    covid_keywords = ('covid','corona', 'virus', 'variant', 'vaccine',
                        'hospital', 'cdc','icu', 'lockdown', 'omicron', 'delta', 
                        'ventilator','infect','mask', 'cases', 'n95')
    initial_series = dataframe.text.str.count(covid_keywords[0], flags=re.IGNORECASE)
    for keyword in covid_keywords[1:]:
        temp_series = dataframe.text.str.count(keyword, flags=re.IGNORECASE)
        initial_series += temp_series
    dataframe['keyword_len'] = initial_series
    # if not regular_news:
    #     dataframe = dataframe[(dataframe.text.str.count('covid', flags=re.IGNORECASE) >= 1) | 
    #                         (dataframe.text.str.count('omicron', flags=re.IGNORECASE) >= 1) | 
    #                         (dataframe.text.str.count('delta', flags=re.IGNORECASE) >= 1)]
    return dataframe


def covid_format_text(dataframe):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe)
    dataframe = dataframe[(((dataframe.keyword_len >= 5) & (dataframe.keyword_len < 40)) & (dataframe.text_len < 500))|
                    ((dataframe.keyword_len >= 7)  & (dataframe.text_len < 750))|
                    (((dataframe.keyword_len >= 10) & (dataframe.text_len <= 1050)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 14) & (dataframe.text_len <= 1500)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 20) & (dataframe.text_len <= 2000)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 30) & (dataframe.text_len <= 3200)) & (dataframe.text_len > 750))]
    return dataframe 

def experimental_covid_article_filtering(dataframe):
    mask = (dataframe.text.str.count('covid', flags=re.IGNORECASE) >= 2) | \
            (dataframe.text.str.count('omicron', flags=re.IGNORECASE) >= 1) | \
            (dataframe.text.str.count('delta', flags=re.IGNORECASE) >= 1)
    dataframe = dataframe[(dataframe.keyword_len >= 5) & (mask)]
    return dataframe 

def headline_formatting(dataframe):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe)
    dataframe = dataframe[(((dataframe.keyword_len >= 5) & (dataframe.keyword_len < 40)) & (dataframe.text_len < 500))|
                    ((dataframe.keyword_len >= 7)  & (dataframe.text_len < 750))|
                    (((dataframe.keyword_len >= 10) & (dataframe.text_len <= 1050)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 14) & (dataframe.text_len <= 1500)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 20) & (dataframe.text_len <= 2000)) & (dataframe.text_len > 750)) |
                    (((dataframe.keyword_len >= 30) & (dataframe.text_len <= 3200)) & (dataframe.text_len > 750))]
    dataframe = dataframe[['article_id', 'date', 'publisher', 'title', 'page_num']]
    return dataframe
    
def regular_news_formatting(dataframe, num_articles_to_sample=150000):
    dataframe = base_text_formatting(dataframe)
    dataframe = count_keywords(dataframe, regular_news=True)
    dataframe = dataframe[(dataframe.text.str.count('covid', flags=re.IGNORECASE) <= 1)]
    dataframe = dataframe[dataframe.keyword_len <= 3]
    dataframe = dataframe.sample(num_articles_to_sample)
    return dataframe