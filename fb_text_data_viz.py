import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize

import string
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from collections import Counter
import en_core_web_sm

# open the file
text = open('filtered_words.txt', mode='r', encoding='utf-8-sig').read()

# split up the words into a list
words = text.split()

### Count Most Used Words ###

def word_count_viz(words_list):

    # create an empty dictionary to store the words
    word_count = {}

    # create a translator using the string function, this will remove any punctuation

    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    for w in words_list:
        w = w.translate(translator).lower()
        if w not in stop_words:
            if w not in word_count:
                word_count[w] = 1
            else:
                word_count[w] += 1

    print(word_count)

    word_count_list = Counter(word_count)
    listy = word_count_list.most_common()

    df = pd.DataFrame(listy, columns=['word', 'count'])
    df2 = df.dropna(axis=0, how='any')

    df2.to_csv('word_counts', index=False)
    print(df2.head(250))
    sns.barplot(x=df2['word'][300:331], y=df2['count'])
    plt.xticks(rotation=-45)
    plt.show()

word_count_viz(words)

### Sentiment Analysis ###

tb_sentiments = []
custom_sentiments = []
vader_sentiments = []

def sentiment_analysis(csv_documemnt):

    sentiment_text = open(csv_documemnt, mode='r', errors='ignore')
    text_read = csv.reader(sentiment_text)

    for row in text_read:

        print("Analyzing sentence...")
        text_blob_analysis = TextBlob(str(row))

        sentiment = text_blob_analysis.sentiment.polarity

        if sentiment >= 0.05:
            tb_sentiments.append('Pos')
        elif sentiment <= 0.049 and sentiment >= -0.049:
            tb_sentiments.append('Unk')
        elif sentiment <= -0.05:
            tb_sentiments.append('Neg')

        analyzer = SentimentIntensityAnalyzer()
        sentiment_3 = analyzer.polarity_scores(str(row))
        if sentiment_3['compound'] >= 0.05:
            vader_sentiments.append('Pos')
        elif sentiment_3['compound'] <= 0.049 and sentiment_3['compound'] >= -0.049:
            vader_sentiments.append('Unk')
        elif sentiment_3['compound'] <= -0.05:
            vader_sentiments.append('Neg')

sentiment_analysis("fb_posts.csv")

tb_sent_df = pd.DataFrame(tb_sentiments, columns=['TB Sentiment'])
vader_sent_df = pd.DataFrame(vader_sentiments, columns=['VD_Sentiment'])
new_df = pd.read_csv('fb_posts.csv', names=['Post'])
new_df.insert(1, "TB_Sent", tb_sent_df)
new_df.insert(2, "VD Sentiment", vader_sent_df)
print(new_df.head())
new_df.to_csv('fb_posts_sentiment5.csv')

def get_sent_counts(input_csv):
    num_pos = 0
    num_neg = 0

    dataframe = pd.read_csv(input_csv)

    for idx, row in dataframe.iterrows():
        if row['TB_Sent'] and row['VD Sentiment'] == 'Pos':
            num_pos += 1
        if row['TB_Sent'] and row['VD Sentiment'] == 'Neg':
            num_neg += 1

    print("Total positive posts:" + str(num_pos))
    print("Total negative posts:" + str(num_neg))

get_sent_counts('fb_posts_sentiment5.csv')

### Create Word Clouds ###

# word cloud bubble

nltk_stop_words = set(stopwords.words('english'))

text2 = open('fb_messages_text.txt', mode='r', encoding='utf-8-sig')
lines = text2.readlines()[:]
text_complete = "".join(lines)
#print(text_complete)

tokens = word_tokenize(text_complete)

def make_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color='black',
        max_words=500,
        max_font_size=40,
        scale=3,
        random_state=9,
        stopwords=nltk_stop_words,
        relative_scaling=0.5
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')

    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

make_wordcloud(text_complete)

### name entity recognition ###

nlp = en_core_web_sm.load()

# make a list of common words/phrases to drop
drop_list = '|'.join(["Stitcher", "Goodreads", "Travel Ninja", 'Stitche'])
text_csv = pd.read_csv('fb_messages.csv', 'r', error_bad_lines=False, names=['Messages'])
text_csv = text_csv[~text_csv.Messages.str.contains(drop_list, na=False)]
#print(text_csv.head(5))

sentences = text_csv['Messages'].tolist()
print(sentences)

doc = nlp(str(sentences))
print([(X.text, X.label_) for X in doc.ents])

labels = [x.label_ for x in doc.ents]
print(Counter(labels))

items = [x.text for x in doc.ents]
print(Counter(items).most_common(20))

def word_counter(doc, ent_name, col_name):
    ent_list = []
    for ent in doc.ents:
        if ent.label_ == ent_name:
            ent_list.append(ent.text)
    df = pd.DataFrame(data=ent_list, columns=[col_name])
    return df

persons_df = word_counter(doc, 'PERSON', 'Named Entities')
#print(persons_df.head(5))

art_df = word_counter(doc, 'WORK_OF_ART', 'Works Of Art')
#print(art_df.head(5))

org_df = word_counter(doc, 'ORG', 'Organizations')
#print(org_df.head(5))

gpe_df = word_counter(doc, 'GPE', 'GPEs')
#print(gpe_df.head(5))

norp_df = word_counter(doc, 'NORP', 'NORPs')
#print(norp_df.head(5))

prod_df = word_counter(doc, 'PRODUCT', 'Products')
#print(prod_df.head(5))

def plot_categories(column, df, num):
    sns.countplot(x=column, data=df,
                  order=df[column].value_counts().iloc[0:num].index)
    plt.xticks(rotation=-90)
    plt.show()

plot_categories("Named Entities", persons_df, 30)
plot_categories("Works Of Art", art_df, 30)
plot_categories("Organizations", org_df, 30)
plot_categories("GPEs", gpe_df, 30)

