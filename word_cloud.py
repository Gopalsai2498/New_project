# -*- coding: utf-8 -*-

###########################
### W O R D   C L O U D ###
###########################

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
from gensim.models import word2vec
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt


infile = 'D:/Sentier/project/'
df = pd.read_csv(infile+'df_cleansed.csv')

comment_words = ''

for char in df.cleansed_description:
    
    # typecaste each val to string 
    char = str(char) 
  
    # split the value 
    tokens = char.split() 
    
    comment_words += " ".join(tokens)+" "

#create document term matrix
cv = CountVectorizer()
data_cv = cv.fit_transform(df.cleansed_description)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm_transp = data_dtm.transpose()

# Find the top 10 words said by each insight
top_dict = {}
for c in data_dtm_transp.columns:
    top = data_dtm_transp[c].sort_values(ascending=False).head(10)
    top_dict[c]= list(zip(top.index, top.values))
#top_dict

# Look at the most common top words --> add them to the stop word list
# Let's first pull out the top 10 words
words = []
for w in data_dtm_transp.columns:
    top = [word for (word, count) in top_dict[w]]
    for t in top:
        words.append(t)
#words

#the most common 5 words
Counter(words).most_common(5)

#add stopwords from infile csv
add_stopword = list(pd.read_csv(infile+"add_stopwords.csv")['0'])
stop_words = text.ENGLISH_STOP_WORDS.union(add_stopword)

#word cloud
wordcloud = WordCloud(
                        background_color='white',
                        stopwords = stop_words,
                        max_words = 30, random_state=1,
                        max_font_size=50).generate(comment_words)

#fig = plt.figure(1)
plt.figure(figsize=[14,7])
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


#######################################
#   W O R D   A S S O C I A T I O N
#######################################

# build a corpus for the word2vec model
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    word_list = data.split(" ")
    return word_list

corpus = df.cleansed_description.apply(build_corpus)

model = word2vec.Word2Vec(corpus, size=100, window=5, min_count=10, workers=4)

#first 5 elements of model vocabulary
[x for x in model.wv.vocab][0:5]

#find word associations
[(item[0],round(item[1],2)) for item in model.most_similar('gilt')]
