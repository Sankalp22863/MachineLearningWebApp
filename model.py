from copyreg import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class HateComments:
    def __init__(self):
        self.data = self.get_data()
        self.tfidf = self.load_tfidf()
        self.model = self.load_model()
        return

    def import_data(self, data):
        return data
    
    def load_tfidf(self):
        file = open("tfidf_pickle_fit","rb")
        tfidf = pickle.load(file)
        file.close()
        return tfidf

    def load_model(self):
        file = open("pickled_logistic.pkl", "rb")
        model = pickle.load(file)
        file.close()
        return model
    
    def prediction(self):
        self.data_transformed = self.data["Comment"]
        self.data_transformed = self.tfidf.transform(self.data_transformed)
        tfidf_vectors = TfidfVectorizer(max_df=0.90, min_df=2, max_features=9000, 
                                        stop_words='english',
                                        ngram_range=(1, 3))
        df_vector = pd.DataFrame(self.data_transformed.todense(),columns = tfidf_vectors.get_feature_names())
        self.pred = self.model.predict(df_vector)
        return self.pred