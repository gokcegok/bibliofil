# -*- coding: utf-8 -*-
"""
Created on Wed May 24 00:36:07 2023

@author: gokcegok
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from string import digits
from alphabet_detector import AlphabetDetector
import lemmatizer
import warnings
import ocrspace

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")


# %% Functions

# accent transform dict
ACCENT_TRANSFORM = {"Î": "İ", "î": "i", "İ": "i", "â": "a", 
                    "Â": "A", "Û": "U", "û": "u"}

ad = AlphabetDetector()

# Turkish Lemmas
with open("./stop-words/tr-lemmas.txt", "r") as file:
    revisedDict = eval(file.read())

# STOP WORDS

with open('./stop-words/ahmetax-tr-stopwords-v2.txt', 'r', encoding="utf-8") as file:
    tr_stop_words = file.readlines()
    
STOP_WORDS = [word.strip("\n") for word in tr_stop_words]

def preprocess_names(string):
    
    string = re.sub(r'[^\w\s]', '', string) 
    # sayilari sil
    remove_digits = str.maketrans('', '', digits)
    string = string.translate(remove_digits)
    # satir sonlarini ve fazladan bosluklari sil
    string = re.sub('\n', '', string)
    string = re.sub('  +', ' ', string)
    # aksanli harfleri aksansizlarina donustur
    string = string.translate(str.maketrans(ACCENT_TRANSFORM))
    string = string.lower()
    
    return string

def preprocess_words(string):
    
    ad = AlphabetDetector()

    # Turkish Lemmas
    with open("./stop-words/tr-lemmas.txt", "r") as file:
        revisedDict = eval(file.read())

    # STOP WORDS

    with open('./stop-words/ahmetax-tr-stopwords-v2.txt', 'r', encoding="utf-8") as file:
        tr_stop_words = file.readlines()
        
    STOP_WORDS = [word.strip("\n") for word in tr_stop_words]
    

    if ad.is_latin(string) == True:
        # noktalama isaretlerini sil
        string = re.sub(r'[^\w\s]', '', string) 
        # sayilari sil
        remove_digits = str.maketrans('', '', digits)
        string = string.translate(remove_digits)
        # satir sonlarini ve fazladan bosluklari sil
        string = re.sub('\n', '', string)
        string = re.sub('  +', ' ', string)
        # aksanli harfleri aksansizlarina donustur
        string = string.translate(str.maketrans(ACCENT_TRANSFORM))
        string = string.lower()
        # remove stop words
        
        row = string.split(" ")
        
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
    
    # dataset: pd.DataFrame
    # var_index: index of book
    
    # TF-IDF
    tfidf = TfidfVectorizer() # tfIdf nesnesi
    
    if len(serie) != 0:
        tfidf_matrix = tfidf.fit_transform(serie)
    else:
        tfidf_matrix = tfidf.fit_transform(dataset["all_info"])

    # Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    similarity_scores = pd.DataFrame(cosine_sim[var_index], columns=["score"])
    top = similarity_scores.sort_values("score", ascending=False)[1:26].index
    recommendations = pd.DataFrame(columns=["score", "num_of_purchasers"])
    recommendations["score"] = similarity_scores.iloc[top]
    recommendations["num_of_purchasers"] = dataset.iloc[top]["number_of_purchasers"]
    recommendations["result"] = recommendations["score"]*0.8 + \
                                recommendations["num_of_purchasers"]*0.2
                                
    mix_top10 = recommendations.sort_values("result", ascending=False)[0:10].index
     
    return mix_top10


# %% Dataset

df_path = "./Datasets/kitapyurdu_data_final_v4.csv"
df = pd.read_csv(df_path)

book_names = pd.Series(df.index, index=df["name"].str.lower())

# %% Recommendation

# if __name__ == "__main__":
    
print("""
      \nKitap Adı: (1)
      \nYazar Adı: Hakkında(2), Eserleri(3)
      \nAnahtar Kelime/Açıklama: (4)
      \nGörüntü: (5)      
      \n--------------------------------------
      """)

selections = [1, 2, 3, 4]
while True:
    
    try:
        
        selection = int(input("\nSeçiminiz: "))
        
        if selection in selections:
            
            search = input("\nArama: ").lower()

            if selection == 1:
                
                # With book name
                try:
                    
                    book_index = book_names[search]
                    recommendation_indices = recommend(df, book_index)
                    break

                except:
                    
                    info = df["all_info"]
                    info = info.append(pd.Series(preprocess_names(search)), 
                                       ignore_index=True)
                    book_index = len(info) - 1    
                    recommendation_indices = recommend(df, book_index, info)
                    break

            elif selection == 2:
                
                # Hakkında
                data = df[~df["author"].str.lower().str.contains(search)]
                info = data["all_info"]
                info = info.append(pd.Series(search), ignore_index=True)
                book_index = len(info) - 1
                recommendation_indices = recommend(data, book_index, info)
                break
                
            elif selection == 3:
                
                # Eserler
                info = df["all_info"]
                info = info.append(pd.Series(search), ignore_index=True)
                book_index = len(info) - 1
                recommendation_indices = recommend(df, book_index, info)
                break

            elif selection == 4:
                
                # With description
                info = df["all_info"]
                info = info.append(pd.Series(search), ignore_index=True)
                book_index = len(info) - 1

                recommendation_indices = recommend(df, book_index, info)
                break
            
        if selection == 5:
            
            # With image
            image_path = input("Görüntü yolu: ")
            file_ext = image_path.split(".")[1]
            
            api = ocrspace.API(endpoint='https://api.ocr.space/parse/image', 
                                     api_key='7c3412d42d88957', 
                                     language=ocrspace.Language.English,
                                     OCREngine=2, filetype=file_ext)
            
            search = api.ocr_file(image_path)
            
            info = df["all_info"]
            info = info.append(pd.Series(search), ignore_index=True)
            book_index = len(info) - 1
        
            recommendation_indices = recommend(df, book_index, info)
            break  
        
        else:
            
            print("\nYanlış giriş........")
            
    except ValueError:
        
        print("\nYanlış giriş........")

recommendations = df.iloc[recommendation_indices]
print(recommendations[["name", "author"]])   


