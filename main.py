import streamlit as st
import streamlit.components.v1 as com
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helpers import recommend_newData, image2text
import re

st.set_page_config(layout="wide")
image = Image.open("webFiles/rahibe-teresa2.jpg")

with open("webFiles/imgStyles.css") as styles:
    design = styles.read()


@st.cache_data
def get_data():

    df = pd.read_csv("dataset/philosophy-books-data-tr-processed.csv")
    df["number_of_purchasers"] = df["number_of_purchasers"].astype("int")
    min_max = MinMaxScaler()
    df["number_of_purchasers"] = min_max.fit_transform(df[["number_of_purchasers"]])

    return df


def home(tab_home):

    description = tab_home.container()
    description.image(image)


def books(tab, data):

    authors = data["author"].unique()
    selected_author = tab.selectbox(label="Yazar Adı: ", options=authors)

    book = data[data["author"] == selected_author]
    selected_book = tab.selectbox(label="Kitap Adı: ", options=book)

    col1, col2 = tab.columns([1, 6])

    with col1:
        com.html(f"""
                    <div>
                    <style>
                    {design}
                    </style>
                    <img class="bookMaxi" src={data[(data["author"] == selected_author) &
                                                    (data["name"] == selected_book)]["image_link"].values[0] 
                                                }> 
                    </div>
                    """)

    col2.markdown(str(data[(data["author"] == selected_author) &
                             (data["name"] == selected_book)]["description"].values[0]))


def get_results(data, recommendation_indices, tab, start=0):

    recommendations = data.iloc[recommendation_indices][start:start + 5]
    one, two, three, four, five = tab.columns(5)

    for index, col in enumerate([one, two, three, four, five]):

        try:
            book = data.loc[data.index == recommendations.index[index], :]
            with col:
                com.html(f"""
                            <div>
                            <style>
                            {design}
                            </style>
                            <img class="bookMini" src={book.image_link.values[0]}> 
                            </div>
                            """)

            col.markdown(f"**{book.name.values[0]}**")
            col.markdown(book.author.values[0])
            col.markdown(book.publisher.values[0])
        except IndexError:
            continue


def recommender(tab, data):

    col1, col2 = tab.columns([2, 5])

    col11, col12 = col1.columns([5, 1])
    uploaded_file = col11.file_uploader("**Görüntü:**" + "\n" + "\n*kitap kapağı ya da iç sayfasıyla arama yapabilirsiniz*", accept_multiple_files=False)
    if uploaded_file is not None:

        search = image2text(uploaded_file)
        recommendation_indices = recommend_newData(data, search)
        get_results(data, recommendation_indices, col2)

    col11, col12 = col1.columns([5, 3])
    search = col11.text_input("", "url:görüntü")
    col12.write("")
    col12.write("")
    if col12.button(':camera_with_flash:'):
        search = image2text(search)
        recommendation_indices = recommend_newData(data, search)
        get_results(data, recommendation_indices, col2)

    search = col11.text_input("**Anahtar Kelime:**", "stoacılık")
    col12.write("")
    col12.write("")
    if col12.button(':female-detective: :male-detective:'):
        recommendation_indices = recommend_newData(data, search)
        col2.subheader("***Arama:*** " + f"*{search[0:120]}*" + "...")
        col2.write("")
        col2.write("")
        get_results(data, recommendation_indices, col2)

    book = data["name"].unique()
    selected_book = col11.selectbox(label="**Kitap Adı:**", options=book)
    col12.write("")
    col12.write("")
    col12.write("")
    if col12.button(':book:'):
        recommendation_indices = recommend_newData(data, selected_book)
        col2.subheader("***Arama:*** " + f"*{selected_book}*")
        col2.write("")
        col2.write("")
        get_results(data, recommendation_indices, col2, 1)

    author = data["author"].unique()

    selected_author = col11.selectbox(label="**Yazar Adı:**", options=author)
    about = col11.checkbox('ilişkili')

    if about:

        # selected_author = preprocess_authorNames(selected_author)
        col12.write("")
        col12.write("")
        if col12.button(':lower_left_fountain_pen:'):

            col2.subheader("***Arama:*** " + f"*{selected_author[0:120]}*" + " ***ile ilişkili***")
            col2.write("")
            col2.write("")

            search = selected_author
            search = re.sub("\n", " ", search)
            search = search.split(" ")
            search = "|".join(search)

            df = data[(data["name"].str.contains(search, regex=True)) |
                      (data["description"].str.contains(search, regex=True))]

            df = df[~df["author"].str.contains(search, regex=True)]

            recommendation_indices = recommend_newData(df, search)

            recommendations = df.iloc[recommendation_indices][0:5]
            one, two, three, four, five = col2.columns(5)

            for index, col in enumerate([one, two, three, four, five]):

                try:
                    book = df.loc[df.index == recommendations.index[index], :]
                    with col:
                        com.html(f"""
                                    <div>
                                    <style>
                                    {design}
                                    </style>
                                    <img class="bookMini" src={book.image_link.values[0]}> 
                                    </div>
                                    """)
                    col.subheader(f"{book.name.values[0]}")
                    col.markdown(book.author.values[0])
                    col.markdown(book.publisher.values[0])
                except IndexError:
                    continue

    else:

        col12.write("")
        col12.write("")
        if col12.button(':lower_left_fountain_pen:'):
            recommendation_indices = data[data["author"] == selected_author].sort_values(by="number_of_purchasers",
                                                                                         ascending=False).index
            col2.subheader("***Arama:*** " + f"*{selected_author[0:140]}*")
            col2.write("")
            col2.write("")
            get_results(data, recommendation_indices, col2)


def main():

    st.title(":green[bibliofil] :books:")

    data = get_data()

    home_tab, books_tab, recommender_tab = st.tabs(["Giriş", "Kitaplar, Kitaplarımız", "Tavsiyeler"])

    # Home Tab
    home(home_tab)

    # Books Tab
    books(books_tab, data)

    # Recommender Tab

    recommender(recommender_tab, data)


if __name__ == "__main__":

    main()
