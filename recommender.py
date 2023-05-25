# -*- coding: utf-8 -*-
"""
Created on Wed May 24 00:36:07 2023

@author: gokcegok
"""

import pandas as pd
import warnings
from helpers import recommend_newData, recommend_fromData, image2text


pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")


# %% Dataset

df_path = "./philosophy-books-data-tr-processed.csv"
df = pd.read_csv(df_path)

# %% Recommendation
    
print("""
      \n--------------------------------------
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
            
            search = input("\nArama: ")

            if selection == 1:
                
                search = search.lower()
                
                # recommend with book name
                try:
                    
                    recommendation_indices = recommend_fromData(df, search)
                    break

                except:
                    
                    recommendation_indices = recommend_newData(df, search)
                    recommendations = df.iloc[recommendation_indices]
                    break

            elif selection == 2:
                
                # books about the author
                
                author = search.split(" ")
                author = "|".join(author)
                data = df[(df["name"].str.contains(author, regex=True)) |
                      (df["description"].str.contains(author, regex=True))]

                data = data[~data["author"].str.contains(author, regex=True)]  
                
                recommendation_indices = recommend_newData(data, search)
                recommendations = data.iloc[recommendation_indices]
                break
            
            
                
            elif selection == 3:
                
                # books of the author
                search = search.lower()  

                recommendation_indices = recommend_newData(df, search)
                recommendations = df.iloc[recommendation_indices]
                recommendation_indices = recommendations[recommendations["author"].str.lower().str.contains(search)].index
                recommendations = df.iloc[recommendation_indices]
                break

            elif selection == 4:
                
                # recommend with description
                search = search.lower()
                recommendation_indices = recommend_newData(df, search)
                recommendations = df.iloc[recommendation_indices]
                break
            
        if selection == 5:
            
            # recommend with image
            image_path = input("Görüntü yolu: ")           
            search = image2text(image_path)           
            recommendation_indices = recommend_newData(df, search)
            recommendations = df.iloc[recommendation_indices]
            break  
        
        else:
            
            print("\nYanlış giriş........")
            
    except ValueError:
        
        print("\nYanlış giriş........")


print(recommendations[["name", "author"]])   