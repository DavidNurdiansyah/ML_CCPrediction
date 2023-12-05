import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# membuat sidebar
app_mode = st.sidebar.selectbox(
    'Select Page', ['Home', 'Grafik', 'Prediction', 'About'])

if app_mode == "Home":
    st.write('<h1 style="text-align: center; font-weight: bold;">Motorcycle Number of CC Prediction</h1>',
             unsafe_allow_html=True)
    st.image('motor.jpg')

    st.write('<p style="text-align: center;">Created by Affan Naufal S, David Nurdiansyah, Hafizh Lukmanul H</p>',unsafe_allow_html=True)

elif app_mode == "Grafik": 
    st.write('<h4 style="text-align: center; font-weight: bold;">Dataset</h4>',
            unsafe_allow_html=True)
    data = pd.read_csv('bikes_data.csv')
    st.dataframe(data)

    st.write('<h4 style="font-weight: bold;">Describe from Dataset</h4>',
            unsafe_allow_html=True)
    st.dataframe(data.describe())

    # Filter data berdasarkan tahun
    data1 = data[data['Year'] > 2013]

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig, ax = plt.subplots(figsize=(12, 6))

    # Membuat countplot dengan Seaborn pada axes yang ditentukan
    sns.countplot(data=data1, x='Company', ax=ax,palette="coolwarm")

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Number of bikes company</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Company')
    ax.set_ylabel('Count')

    # Rotasi label pada sumbu x pada axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)


    #Data sorting
    top_10_most_horsepower = data.sort_values(by='Horsepower', ascending=False).head(15)
    top_10_most_cc = data.sort_values(
        by='Number of cc', ascending=False).head(15)
    top_10_least_horsepower = data.sort_values(by='Horsepower', ascending=True).head(15)
    top_10_least_cc = data.sort_values(
        by='Number of cc', ascending=True).head(15)

    # Membuat figure dan axes
    fig1, ax = plt.subplots(figsize=(15, 10))

    # Membuat scatter plot
    sns.scatterplot(data=top_10_most_horsepower, x='Horsepower',
                    y='Number of cc', hue='Company', ax=ax)

    # Menambahkan judul dan label
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Bikes with the Most Horsepower</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Horsepower')
    ax.set_ylabel('Number of cc')

    # Menampilkan plot di Streamlit
    st.pyplot(fig1)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig2, ax = plt.subplots(figsize=(20, 10))

    # Membuat scatter plot pada axes yang ditentukan
    sns.scatterplot(data=top_10_most_cc, x='Horsepower',
                    y='Number of cc', hue='Company', ax=ax)

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Bikes with the Most Number of cc</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Horsepower')
    ax.set_ylabel('Number of cc')

    # Menampilkan plot
    st.pyplot(fig2)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig3, ax = plt.subplots(figsize=(25, 10))

    # Membuat scatter plot pada axes yang ditentukan
    sns.scatterplot(data=top_10_least_horsepower, x='Horsepower',
                    y='Number of cc', hue='Company', ax=ax)

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Bikes with the Least Number of cc</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Horsepower')
    ax.set_ylabel('Number of cc')

    # Menampilkan plot
    st.pyplot(fig3)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig4, ax = plt.subplots(figsize=(20, 10))

    # Membuat scatter plot pada axes yang ditentukan
    sns.scatterplot(data=top_10_least_cc, x='Horsepower',
                    y='Number of cc', hue='Company', ax=ax)

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Bikes with the Least Number of cc</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Horsepower')
    ax.set_ylabel('Number of cc')

    # Menampilkan plot
    st.pyplot(fig4)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig5, ax = plt.subplots(figsize=(12, 14))

    # Membuat countplot pada axes yang ditentukan
    sns.countplot(data=data, y='Company', ax=ax, palette='coolwarm')

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Number of Bikes Company</h4>',
            unsafe_allow_html=True)

    ax.set_xlabel('Count')
    ax.set_ylabel('Company')

    # Menampilkan plot
    st.pyplot(fig5)

    data['Price (in INR)'] = data['Price (in INR)'].str.extract(
        '(\d+)').astype(int)

    top_10_cheapest = data.sort_values(by='Price (in INR)').head(10)

    top_10_costliest = data.sort_values(
        by='Price (in INR)', ascending=False).head(10)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig6, ax = plt.subplots(figsize=(12, 6))

    # Membuat barplot pada axes yang ditentukan
    sns.barplot(data=top_10_cheapest, x='Model', y='Price (in INR)', ax=ax ,palette='coolwarm')

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Cheapest Bikes</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Model')
    ax.set_ylabel('Price (in INR)')

    # Rotasi label pada sumbu x pada axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Menampilkan plot
    st.pyplot(fig6)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig7, ax = plt.subplots(figsize=(12, 6))

    # Membuat barplot pada axes yang ditentukan
    sns.barplot(data=top_10_costliest, x='Model', y='Price (in INR)', ax=ax, palette='coolwarm')

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Top 10 Cosliest Bikes</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Model')
    ax.set_ylabel('Price (in INR)')

    # Rotasi label pada sumbu x pada axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Menampilkan plot
    st.pyplot(fig7)
    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig8, ax = plt.subplots(figsize=(40, 40))

    # Membuat pie chart pada axes yang ditentukan dengan warna
    data['Body Type'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True,
                                            ax=ax, colors=plt.cm.Paired.colors)

    # Menambahkan judul dan menghilangkan label y-axis
    st.write('<h4 style="text-align: center; font-weight: bold;">Distribution of Bike Body Types</h4>',
            unsafe_allow_html=True)
    ax.set_ylabel('')

    # Menampilkan plot
    st.pyplot(fig8)

    # Membuat figure dan axes dengan ukuran yang ditentukan
    fig9, ax = plt.subplots(figsize=(10, 6))

    # Membuat countplot pada axes yang ditentukan
    sns.countplot(data=data, x='Year', ax=ax, palette="coolwarm")

    # Menambahkan judul dan label pada axes
    st.write('<h4 style="text-align: center; font-weight: bold;">Number of Bikes by Year</h4>',
            unsafe_allow_html=True)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')

    # Rotasi label pada sumbu x pada axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Menampilkan plot
    st.pyplot(fig9)

elif app_mode == "Prediction":
    # Membaca dataset
    st.write('<h4 style="text-align: center; font-weight: bold;">Dataset</h4>',
            unsafe_allow_html=True)
    data = pd.read_csv('fixbanget.csv') 
    st.dataframe(data)
    

    # Memilih fitur (features) dan target (label)
    X = data[['Number of Cylinders', 'Horsepower']]
    y = data['Number of cc']

    # Memisahkan data menjadi data latih (training) dan data uji (testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Membuat model Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    st.title('Prediksi CC Motor')

    silinder = st.number_input('Jumlah Silinde Motor : ', 0, 10000000)
    horsepower = st.number_input('Horsepower Motor : ', 0, 10000000)

    if st.button('Prediksi'):
        new_data = pd.DataFrame({'Number of Cylinders': [silinder], 'Horsepower': [horsepower]})
        predicted_cc = model.predict(new_data)
        st.markdown(f'CC motor dengan {silinder} Silinder dan {horsepower} Horsepower : {predicted_cc[0]}') 

elif app_mode == "About":
    def about_page():
        st.markdown("<br>",unsafe_allow_html=True)
        st.write('<h1>Motorcyle Number of CC Prediction</h1>',unsafe_allow_html=True)
        st.image('motor.jpg')
        with open("deskripsi.txt", "r") as f:
            data = f.read()
        st.write(data)

    about_page()
