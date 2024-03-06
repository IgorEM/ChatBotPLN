import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import io
import re
import spacy


data = pd.read_csv("C:/Users/igorm/Desktop/chatBot/dados/lyrics-data.csv")
data1 = pd.read_csv("C:/Users/igorm/Desktop/chatBot/dados/artists-data.csv")

#print("data",data)
#print("data1",data1)



# Renomeando a coluna "ALink" do DataFrame "data" para "Link"
data.rename(columns={'ALink': 'Link'}, inplace=True)

# Unindo os dois DataFrames com base na coluna "Link"
merged_data = pd.merge(data, data1, on='Link')

#print(merged_data)

#Alterando o nome das colunas
merged_data.rename(columns={'Link': 'LinkDoArtista', 'SName': 'NomeDaMusica', 'SLink': 'LinkDaMusica','Lyric': 'LetraDaMusica', 'language': 'Idioma','Artist': 'NomeDoArtista' ,'Genres': 'GeneroMusical','Songs': 'QuantidadeDeMusicas', 'Popularity': 'PopularidadeDoArtista'  }, inplace=True)
#print(merged_data.columns)


#Quero só músicas em Português
merged_data = merged_data[merged_data['Idioma'] == 'pt']


#Vendo a quantidade de linhas e colunas
#print("Número de linhas e Colunas:", merged_data.shape) #(156941, 9)


#tirando \n da letra da musica
merged_data['LetraDaMusica'] = merged_data['LetraDaMusica'].str.replace('\n',' ')

#Botando link completo da coluna LinkDoArtista
merged_data['LinkDoArtista'] = merged_data['LinkDoArtista'].apply(lambda x: "https://www.vagalume.com.br" + x)

#Botando link completo da coluna LinkDaMusica
merged_data['LinkDaMusica'] = merged_data['LinkDaMusica'].apply(lambda x: "https://www.vagalume.com.br" + x)

# Definindo a opção display.max_columns para None para exibir todas as colunas
pd.set_option('display.max_columns', None)

#printando a posiçao 10.000
#print(merged_data.iloc[10000])



#deixando letras, nome da musica e nome do artista menusculos
merged_data['LetraDaMusica'] = merged_data['LetraDaMusica'].str.lower()
merged_data['NomeDoArtista'] = merged_data['NomeDoArtista'].str.lower()
merged_data['NomeDaMusica'] = merged_data['NomeDaMusica'].str.lower()



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#
#tokenizar letras e nomesdemusicas e nome do artista
merged_data['LetraDaMusica'] = merged_data['LetraDaMusica'].apply(word_tokenize)
merged_data['NomeDoArtista'] = merged_data['NomeDoArtista'].apply(word_tokenize)
merged_data['NomeDaMusica'] = merged_data['NomeDaMusica'].apply(word_tokenize)

#retirar StopWords
stop_words = set(stopwords.words('portuguese'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

merged_data['LetraDaMusica'] = merged_data['LetraDaMusica'].apply(remove_stopwords)
merged_data['NomeDoArtista'] = merged_data['NomeDoArtista'].apply(remove_stopwords)
merged_data['NomeDaMusica'] = merged_data['NomeDaMusica'].apply(remove_stopwords)

#lematizar 


# Criar uma instância de WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

merged_data['LetraDaMusica'] = merged_data['LetraDaMusica'].apply(lemmatize)
merged_data['NomeDoArtista'] = merged_data['NomeDoArtista'].apply(lemmatize)
merged_data['NomeDaMusica'] = merged_data['NomeDaMusica'].apply(lemmatize)

print(merged_data.iloc[1])

merged_data.to_csv("LetrasDasMusicas_PreProcessados.csv", index=False)
