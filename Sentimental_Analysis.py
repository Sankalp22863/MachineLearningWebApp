import numpy as np
import pandas as pd

from textblob import TextBlob

import subprocess


class Senti_Analysis:
    def __init__(self):
        subproc = subprocess.Popen(
            "pip3 install textblob", shell=True, stdout=subprocess.PIPE)
        subproc = subprocess.Popen(
            "python -m textblob.download_corpora", shell=True, stdout=subprocess.PIPE)
        subprocess_return = subproc.stdout.read()
        # print(subprocess_return)
        self.Polarity = []
        self.Subjectivity = []
        return

    def SentimientAnalysis(self, df):
        # Defining the arrays for the Polarity and the Subjectivity of the comment.
        # print(df.shape[0])
        # Now Moving through each comment and then finding its polarity and subjectivity.
        for i in range(df.shape[0]):
            text = df.iloc[i]['Comment']
            # print(i, text)

            blob = TextBlob(text)

            blob.tags

            blob.noun_phrases

            polarity = []
            subjectivity = []

            for sentence in blob.sentences:
                polarity.append(sentence.sentiment.polarity)
                subjectivity.append(sentence.sentiment.subjectivity)

            self.Polarity.append(sum(polarity)/len(polarity))
            self.Subjectivity.append(sum(subjectivity)/len(subjectivity))

        # Now appending the values to the dataframe.
        df = self.append_vals(df)
        return df

    def append_vals(self, df):
        df["Polarity"] = self.Polarity
        df['Subjectivity'] = self.Subjectivity
        return df


Senti_Analysis()
