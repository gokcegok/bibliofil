# -*- coding: utf-8 -*-
"""
Created on Thu May 25 00:02:20 2023

@author: gokcegok
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from string import digits
from alphabet_detector import AlphabetDetector
import lemmatizer
import ocrspace
import pandas as pd
import warnings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")


# %% ATTRIBUTES

# accent transform dict
ACCENT_TRANSFORM = {"Î": "İ", "î": "i", "İ": "i", "â": "a", 
                    "Â": "A", "Û": "U", "û": "u"}

# Turkish Lemmas
turkish_lemmas_path = "./stop-words/tr-lemmas.txt"
with open(turkish_lemmas_path, "r") as file:
    revisedDict = eval(file.read())

# Stop Words
stop_words_path = './stop-words/ahmetax-tr-stopwords-v2.txt'
with open(stop_words_path, 'r', encoding="utf-8") as file:
    tr_stop_words = file.readlines()
    
STOP_WORDS = [word.strip("\n") for word in tr_stop_words]


# %% FUNCTIONS


def preprocess_names(string):
    
    """
    Preprocessing book names.

    Parameters
    ----------
    string : book name
    Returns
    -------
    string : modified book name

    """
    
    # remove punctuation
    string = re.sub(r'[^\w\s]', '', string) 
    # remove digits
    remove_digits = str.maketrans('', '', digits)
    string = string.translate(remove_digits)    
    # remove line breaks and extra spaces   
    string = re.sub('\n', '', string)
    string = re.sub('  +', ' ', string)
    # remove letter accents
    string = string.translate(str.maketrans(ACCENT_TRANSFORM))
    string = string.lower()
    
    return string


def preprocess_authorNames(string):
    
    """
    Preprocessing author names.

    Parameters
    ----------
    string : author name
    Returns
    -------
    string : modified author name

    """
    
    # remove line breaks and extra spaces
    print(string)
    string = re.sub("[\n]", '', string)
    string = string.replace("\n", "")
    print(string)
    # remove punctuation
    string = re.sub(r'[^\w\s]', '', string)
    print(string)    
   
    return string.lower()
    

def preprocess_words(string):
    
    """
    Preprocessing descriptions about books.

    Parameters
    ----------
    string : description(includes all information about a book)

    Returns
    -------
    string : modified description

    """

    # Turkish Lemmas
    with open("./stop-words/tr-lemmas.txt", "r") as file:
        revisedDict = eval(file.read())

    # STOP WORDS
    with open('./stop-words/ahmetax-tr-stopwords-v2.txt', 'r', encoding="utf-8") as file:
        tr_stop_words = file.readlines()
        
    STOP_WORDS = [word.strip("\n") for word in tr_stop_words]
    
    ad = AlphabetDetector()  # for handling non-latin alphabets
    if ad.is_latin(string) == True:
        # remove punctuation
        string = re.sub(r'[^\w\s]', '', string) 
        # remove digits
        remove_digits = str.maketrans('', '', digits)
        string = string.translate(remove_digits)
        #  remove line breaks and extra spaces  
        string = re.sub('\n', '', string)
        string = re.sub('  +', ' ', string)
        # remove letter accents
        string = string.translate(str.maketrans(ACCENT_TRANSFORM))
        string = string.lower()
              
        row = string.split(" ")  # for generating new string
        
        # remove stop words
        for word in row:
            if word in STOP_WORDS:
                row.remove(word)
        
        # lemmatize
        for index, word in enumerate(row):
            lemma = lemmatizer.findPos(word.lower(), 
                                       revisedDict)[0][0].split("_")[0]
            row[index] = lemma
            
        # remove stop words
        for word in row:
            if word in STOP_WORDS:
                row.remove(word)
        
        string = " ".join(row)
        
    else:
        string = " "
    
    return string


def recommend(dataset, var_index, serie=pd.Series()):
    
    """
    This function returns recommended indices for for the relevant book.

    Parameters
    ----------
    dataset : pd.DataFrame
    var_index : int
                index of book
    serie : pd.Series
            if there is no new data, serie is a empty pandas serie as default. 
            else, "serie" is created by adding new data to the dataset.

    Returns
    -------
    mix_top10 : pandas.Index
                indices of top 10 recommendations
    """
    
    # TF-IDF
    tfidf = TfidfVectorizer()  # tfIdf object
    
    if len(serie) != 0:
        tfidf_matrix = tfidf.fit_transform(serie)
    else:
        tfidf_matrix = tfidf.fit_transform(dataset["all_info"])

    # Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    similarity_scores = pd.DataFrame(cosine_sim[var_index], columns=["score"])
    
    recommendations = pd.DataFrame(columns=["score", "num_of_purchasers"])
    
    # most similar 25 books
    top = similarity_scores.sort_values("score", ascending=False)[1:21].index
            
    recommendations["score"] = similarity_scores.iloc[top]    
    recommendations["num_of_purchasers"] = dataset.iloc[top]["number_of_purchasers"]   
    
    # result score = 70% of cosine similarity score + 30% of number of purchasers  
    recommendations["result"] = recommendations["score"]*0.7 + \
                                recommendations["num_of_purchasers"]*0.3
                                
    # The index of the 10 books with the highest "result" score        
    mix_top8 = recommendations.sort_values("result", ascending=False)[0:10].index
     
    return mix_top8


def recommend_fromData(dataset, search):
    """   
    This function generates recommendation for data in dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
    search : string
    Returns
    -------
    recommendation_indices : pandas.Index
    """
    
    book_names = pd.Series(dataset.index, index=dataset["name"].str.lower())
    book_index = book_names[search]
    recommendation_indices = recommend(dataset, book_index)
    
    return recommendation_indices


def recommend_newData(dataset, search):
    """   
    This function generates recommendation for new data.

    Parameters
    ----------
    dataset : pandas.DataFrame
    search : string
    Returns
    -------
    recommendation_indices : pandas.Index
    """
    
    info = dataset["all_info"]
    info = info.append(pd.Series(preprocess_names(search)), 
                       ignore_index=True)
    book_index = len(info) - 1    
    recommendation_indices = recommend(dataset, book_index, info)
    
    return recommendation_indices


def recommend_aboutAuthor(dataset, search):
    
    """   
    This function generates recommendations for books
    about the given author.

    Parameters
    ----------
    dataset : pandas.DataFrame
    search : string 
             author name
    Returns
    -------
    recommendation_indices : pandas.Index
    """

    indexAuthor = dataset[dataset['author'] == search].index
    dataset = dataset.drop(indexAuthor)
    
    info = dataset["all_info"]
    info = info.append(pd.Series(preprocess_names(search)), 
                       ignore_index=True)
    book_index = len(info) - 1    
    
    recommendation_indices = recommend(dataset, book_index, info)
    
    return recommendation_indices


def image2text(image_path):
    """
    This function returns text in the given image.

    Parameters
    ----------
    image_path : absolute path of the image (book cover)

    Returns
    -------
    text : the text in the given image

    """

    api = ocrspace.API(endpoint='https://api.ocr.space/parse/image', 
                             api_key='7c3412d42d88957', 
                             language=ocrspace.Language.English,
                             OCREngine=2)
    
    try:
    
        text = api.ocr_file(image_path)         
    except:    
        
        try:
            text = api.ocr_url(image_path)
            
        except:

                text = api.ocr_base64(image_path)
    
    return text