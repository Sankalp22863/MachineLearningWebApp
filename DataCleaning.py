# importing the important library
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
import re
import seaborn as sns
from googletrans import Translator, constants
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from deep_translator import GoogleTranslator
import emoji

import pickle



class DataCleaning:
    def __init__(self):
        nltk.download('vader_lexicon')
        nltk.download('wordnet')
        nltk.download('stopwords')
        return
    
    def userid(self, tweet):
        # This function calculates the number of userids in the tweets.
        count = 0
        for i in tweet.split():
            if i[0] == '@':
                count += 1
        return count

    def profanity_vector(self, tweet):
        
        # This functions calculates the profanity vector for a given tweet.
        
        bad_words = pd.read_csv('Hinglish_Profanity_List.csv', engine='python', sep=",", encoding='cp1252', header=None)
        bad_words.columns = ['Hinglish', 'English', 'Level']
        hinglish = bad_words['Hinglish'].values
        level = bad_words['Level'].values
        PV = [0] * len(level)
        for word in tweet.split():
            if word in hinglish:
                idx = np.where(hinglish == word)
                PV[level[idx][0]] = 1
        return PV

    def translation(self, tweet):
        
        # Translate the Given text to English.
        translator = GoogleTranslator(target = 'en')
        try:
            trans = translator.translate(tweet)
        except:
            trans = tweet
        trans_tweet = trans
        # print(tweet, trans_tweet)
        if trans_tweet is None:
            trans_tweet = tweet
        
        return trans_tweet.lower()

    def stopword(self, data):
        
        # This function removes the stopwords from the given sentence
        clean = []
        stop_words = set(STOPWORDS)
        
        for tweet in data:
            sentence = []
            for word in tweet.split():
                if word not in stop_words:
                    sentence.append(word)
            clean.append(sentence)
        return clean

    def Lemmatizer(self, tweet):
        
        # This function uses NLTK lemmatization method and clean the sentence.
        lemma = []
        lemmatizer = WordNetLemmatizer()
        
        for word in tweet:
            sentence = []
            for i in word:
                sentence.append(lemmatizer.lemmatize(i))
            lemma.append(' '.join(sentence))
        return lemma

    def SID(self, tweet):
        
        # This function calculates the NLTK sentiments and return the negative, neutral, postive and compound values.
        negative = []
        neutral = []
        positive = []
        compound = []
        
        sid = SentimentIntensityAnalyzer()
        sentiment_score = sid.polarity_scores(tweet)
        
        negative.append(sentiment_score['neg'])
        neutral.append(sentiment_score['neu'])
        positive.append(sentiment_score['pos'])
        compound.append(sentiment_score['compound'])
        
        return negative, neutral, positive, compound


    def cleaning(self, data):
        
        # This functions clean the input text.
        
        user_ids = []
        clean_data_hinglish = []
        clean_translated_data = []
        prof_vector = []
        
        for tweet in tqdm(data):
            userids = self.userid(tweet)
            clean_text = []
            tweet = re.sub(r'\\n', ' ', tweet)  # replacing '\\n' with a space
            tweet = re.sub(r',', ' ', tweet)    # replacing ','  with a space
            tweet = re.sub(r'RT|rt', '', tweet)
            
            for word in tweet.split():
                if word[0] == '@':              # removing user_ids 
                    clean_word = re.sub(word, 'username', word)
                else:
                    clean_word = word.lower()       # lowercase all the words
                    clean_word = re.sub(r'^#\w+', ' ', clean_word)
                    #clean_word = re.sub(r'^\\[a-z0-9].*\\[a-z0-9{3}+]*[^\\n]$', '', clean_word)   # removing emotions in unicode
                    clean_word = re.sub(r'\\', ' ', clean_word)
                    clean_word = re.sub(r'^https:[\a-zA-Z0-9]+', '', clean_word)              # replacing url link with 'url'
                    #clean_word = re.sub(r'[^a-z].\w+', '', clean_word)           # removing evering thing except a-z
                    clean_word = re.sub(r'[!,.:_;$%^\'\#"&]', '', clean_word)
                    clean_text.append(clean_word)
                    
            clean_text = (' ').join(clean_text)
        
            PV = self.profanity_vector(clean_text)  # calling profanity_vector function
            translated_tweet = self.translation(clean_text)  #calling translated_tweet function
            
            user_ids.append(userids)
            clean_data_hinglish.append(clean_text)
            clean_translated_data.append(translated_tweet)
            prof_vector.append(PV)
            
            
        clean_data_hinglish = np.asarray(clean_data_hinglish)
        user_ids = np.asarray(user_ids).reshape(-1,1)
        prof_vector = np.asarray(prof_vector)
        clean_translated_data = np.asarray(clean_translated_data)

            
        return clean_data_hinglish, user_ids, prof_vector, clean_translated_data
    
    def feature_process(self, clean_data, user_ids_train, PV_train):
        ''' This function except the clean data and return Train and Test dataset after stacking userids, profanity vector, negative sentiment, neutral sentiment, 
                        positive sentiment, compound sentiment, n-grams and tfidf features'''
        
        vectorizer = CountVectorizer()
        tfidf = TfidfVectorizer()
        negative = []
        neutral = []
        positive = []
        compound = []

        for comment in clean_data:
            neg, neu, pos, comp = self.SID(comment)
            negative.append(neg), neutral.append(neu), positive.append(pos), compound.append(comp)
        
        clean_data_SW = self.stopword(clean_data)
        
        clean_data_lemm = self.Lemmatizer(clean_data_SW)
        
        vectorizer.fit(clean_data_lemm)
        tfidf.fit(clean_data_lemm)
        
        n_grams = vectorizer.transform(clean_data_lemm)
        tfidf_ngrams = tfidf.transform(clean_data_lemm)
        
        
        negative = np.asarray(negative)
        neutral = np.asarray(neutral)
        positive  = np.asarray(positive)
        compound = np.asarray(compound)
        
        dataset = hstack((user_ids_train, PV_train, negative, neutral, positive, compound, n_grams, tfidf_ngrams))
        
        return dataset

    def replace_emoji_with_text(text):
        text = emoji.demojize(text, delimiters=("", "")) 
        return text

    def data_cleaning(self, project_data):
        X = project_data['Comment']

        clean_data_hinglish, user_ids, prof_vector, clean_translated_data = self.cleaning(X)

        Cleaned_Data = self.feature_process(clean_translated_data, user_ids, prof_vector)

        with open('Model_Pickle_Jar/Dicision_Tree_Pickle.pkl', 'rb') as model:
            decision_tree = pickle.load(model)

        preds = decision_tree.predict(Cleaned_Data)

        return preds