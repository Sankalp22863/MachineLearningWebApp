from xml.etree.ElementTree import Comment
import numpy as np
import pandas as pd
from nltk.stem import *
import re
from sklearn.externals import joblib
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer as SIA

class classifier_abusive:
    def __init__(self, inp_data, labelled_data):
        self.text = pd.read_csv(inp_data)
        self.labelledText = pd.read_csv(labelled_data)


    def customized_preprocess(inp_text):
        url_detect = '/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm'
        mentiontag_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        parsed_text = re.sub('\s+', ' ', inp_text)
        parsed_text = re.sub(url_detect, '<URL>', parsed_text)
        parsed_text = re.sub(mentiontag_regex, '<MENTION>', parsed_text)
        parsed_text = re.sub(hashtag_regex, '<HASHTAG>', parsed_text)
        return parsed_text


    def tokenization(inp_text):
        return (" ".join(re.split("[^a-zA-Z]*", inp_text.lower())).strip())
    
    def lemmatization(inp_text):
        stemmers = [PorterStemmer(),SnowballStemmer(),LancasterStemmer(),RegexpStemmer()]
        stemmer = stemmers[0]
        return [stemmer.stem(t) for t in inp_text.split()]
    
    def get_pos_tags(comments):
        comment_tags = []
        for c in comments:
            tokens = tokenization(customized_preprocess(c))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            comment_tags.append(tag_str)
        return comment_tags

    def other_features_(comment):
        sentiment_analyzer = SIA()
        sentiment = sentiment_analyzer.polarity_scores(comment)

        words = customized_preprocess(comment) #Get text only

        syllables = textstat.syllable_count(words) #count syllables in words
        num_chars = sum(len(w) for w in words) #num chars in words
        num_chars_total = len(comment)
        num_terms = len(comment.split())
        num_words = len(words.split())
        avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
        num_unique_terms = len(set(words.split()))

        ###Modified FK grade, where avg words per sentence is just num words/1
        FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
        ##Modified FRE score, where sentence fixed to 1
        FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

        features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                    num_unique_terms, sentiment['compound']]
        #features = pandas.DataFrame(features)
        return features

    def get_oth_features(tweets):
        """Takes a list of tweets, generates features for
        each tweet, and returns a numpy array of tweet x features"""
        feats=[]
        for t in tweets:
            feats.append(other_features_(t))
        return np.array(feats)

    def transformation(inp_comments, tf_vectorizer, idf_vector, pos_vectorizer):
        tf_array = tf_vectorizer.fit_transform(inp_comments).toarray()
        tfidf_array = tf_array*idf_vector
        print("Built TF-IDF array")

        pos_tags = get_pos_tags(inp_comments)
        pos_array = pos_vectorizer.fit_transform(pos_tags).toarray()
        print("Built POS array")

        oth_array = get_oth_features(inp_comments)
        print("Built other feature array")

        M = np.concatenate([tfidf_array, pos_array, oth_array],axis=1)
        return pd.DataFrame(M)

    def predictions(X, model):
        y_preds = model.predict(X)
        return y_preds

    def class_to_name(class_label):
        if class_label == 0:
            return "Hate speech"
        elif class_label == 1:
            return "Offensive language"
        elif class_label == 2:
            return "Neither"
        else:
            return "No class"

    
    def classify_comments(comments):
        fixed_comments = []
        for i, t_orig in enumerate(comments):
            s = t_orig
            try:
                s = s.encode("latin1")
            except:
                try:
                    s = s.encode("utf-8")
                except:
                    pass
            if type(s) != unicode:
                fixed_comments.append(unicode(s, errors="ignore"))
            else:
                fixed_comments.append(s)
        assert len(comments) == len(fixed_comments)
        comments = fixed_comments
        print(len(comments))

        print("Loading trained classifier... ")
        model = joblib.load('final_model.pkl')

        print("Loading other information...")
        tf_vectorizer = joblib.load('final_tfidf.pkl')
        idf_vector = joblib.load('final_idf.pkl')
        pos_vectorizer = joblib.load('final_pos.pkl')
        #Load ngram dict
        #Load pos dictionary
        #Load function to transform data

        print("Transforming inputs...")
        X = transformation(comments, tf_vectorizer, idf_vector, pos_vectorizer)

        print("Running classification model...")
        predicted_class = predictions(X, model)

        return predicted_class

    def classify(self):
        inp_comments = self.text.Text
        inp_comments = [x for x in inp_comments if type(x) == str]
        train_preds = classify_comments(inp_comments)

        print ("Printing predicted values: ")
        for i,c in enumerate(inp_comments):
            print(c)
            print (class_to_name(train_preds[i]))

        print("Calculate accuracy on labeled data")
        df = pd.read_csv('../data/labeled_data.csv')
        test_comments = self.labelledText.values
        test_comments = [x for x in test_comments if type(x) == str]
        comments_class = test_comments['class'].values
        test_preds = classify_comments(test_comments)
        right_count = 0
        for i,t in enumerate(test_comments):
            if comments_class[i] == test_preds[i]:
                right_count += 1

        accuracy = right_count / float(len(df))
        print("accuracy", accuracy)
