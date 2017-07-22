


import json
import os
import pandas as pd
import numpy as np
import json
import watson_developer_cloud.natural_language_understanding.features.v1 as Features
import pdfminer
import textract
import re
import glob


from flask import jsonify
from scipy.spatial.distance import cdist
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import hierarchical, KMeans, MeanShift
from sklearn.model_selection import train_test_split
from watson_developer_cloud import NaturalLanguageUnderstandingV1

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')



from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import sys

def pdf_parser(data, count):

    fp = file(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    file_name = "Output" + str(count) + ".txt"
    data = data.decode('utf-8').encode('ascii', 'ignore')
    data = data.lower()
    text_file = file(file_name, "w")
    text_file.write(data)
    text_file.close()
    # print data
    return data

# Loops to create txt files for pdfs that are lowered and encoded ascii
def pdf_loop(directory, count):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = directory + "\\" +filename
            pdf_parser(path, count)
            print(count)
            count += 1
            
            continue
        else:
            continue



# Reads text files. Loops directory and appends text to list
articles = []
def article_list(directory):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = directory + "\\" + filename
            text = open(path, 'r').read()
            articles.append(text)
            continue
        else:
            continue
    return articles


articles = article_list(r"C:\Users\austi\Google Drive\PycharmProjects\Debate-Bot\Watson-nlp\articles-text")
#for article in range(1, 80):
 #   file_path = 'articles/test' + str(article) + ".pdf"
#    t = pdfparser(file_path, article)
 #   #t = t.decode('utf-8').encode('ascii', 'ignore')
    #t = t.lower()
# articles.append(t)


print(articles[5])


# In[17]:

def watson_text_analysis(text_):
    natural_language_understanding = NaturalLanguageUnderstandingV1(
      username="8fe00c1b-54ef-4291-a754-cbfcabecacfa",
      password="TguupnbIZO1Z",
      version="2017-02-27")
    # Limits to 2000 keywords, no emotion or sentiment
    response = natural_language_understanding.analyze(
      text=text_,
      features=[
        Features.Keywords(
          emotion=False,
          sentiment=False,
          limit=300
        )
      ]
    )
    my_dump = json.dumps(response, indent=2)
    my_load = json.loads(my_dump)
    new_dict = dict()
    for d in my_load['keywords']:
        lst = [k for k in d.keys()]
        for key in lst:
            new_dict[d[lst[1]]] = d[lst[0]]
    return new_dict


dictionary_list = []
for article in articles:
    d = watson_text_analysis(article)
    dictionary_list.append(d)



# Vocab set created from all articles
key_list = []
for article in dictionary_list:
    key_list.extend([k for k in article.keys()])
    
key_list=list(set(key_list))    





#dictionary_list = [d1, d2, d3]
new_list = []
# Adds 0's for words not in article but in vocabulary
for d in dictionary_list:
    e = dict()
    for key in key_list:
        if key in d.keys():
            e[key] = d[key]
        else:
            e[key] = 0.0
    new_list.append(e)


# saves datafram to csv
df = pd.DataFrame(new_list)
df.to_csv('data.csv', index=False)
#Each article is a row. Each column is a dictionary key from our vocab.

# reads csv from where you want to save it
df = pd.read_csv(r'C:\Users\austi\Documents\data.csv')


# Prints only keywords that add up from each column up to .8. Basically removes less relevent scores that add up to little
for key in new_list[0].keys():
    if new_list[0][key] + new_list[1][key] + new_list[2][key] > .80:
        print key, new_list[0][key], new_list[1][key], new_list[2][key]
