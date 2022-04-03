import enum
import classifier_abusive

import numpy as np
import pandas as pd

classifier_abusive = classifier_abusive.classifier_abusive()

df = pd.read_csv("Data.csv")

comments_data = df.Comment

comments_data = [x for x in comments_data if type(x) == str]

comments_predictions = classifier_abusive.classify_comments(comments_data)

preds = []

# Now getting the final predictions.
for i, t in enumerate(comments_data):
    preds.append(classifier_abusive.class_to_name(comments_predictions[i]))


print(preds)