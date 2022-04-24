import streamlit as st

# Displaying the Words.
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Load Packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# importing the important library
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from scipy.sparse import load_npz


def plot_confusion_matrix(test_y, predict_y):
    
    '''This function returns confusion matrix, precison matrix and recall matrix for 3 class classification'''
    
    C = confusion_matrix(test_y, predict_y)
    print("Number of misclassified points ",(len(test_y)-np.trace(C))/len(test_y)*100)
    
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    
    labels = ['Non Offensive', 'Hate Speech', 'Abusive']
    cmap=sns.light_palette("yellow")
    
    # representing A in heatmap format
    print("-"*50, "Confusion matrix", "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.savefig('Confusion_Matrix_1.png')
    
    print("-"*50, "Precision matrix", "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.savefig('Confusion_Matrix_2.png')

    
    # representing B in heatmap format
    print("-"*50, "Recall matrix" , "-"*50)
    plt.figure(figsize=(10,5))
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.savefig('Confusion_Matrix_3.png')


# Beatufying the Project.
st.set_page_config(page_title="Youtube Universe", page_icon="🖖")

def main():

    project_data = pd.read_csv("ModelData/Project_data.csv")
    # X = project_data['tweet']
    # y = project_data['label']

    X_train = pd.read_csv("ModelData/X_train.csv")
    X_test = pd.read_csv("ModelData/X_test.csv")
    y_train = pd.read_csv("ModelData/y_train.csv")
    y_test = pd.read_csv("ModelData/y_test.csv")


    X_train = X_train.drop("Unnamed: 0", axis = 1)["Comment"]
    X_test = X_test.drop("Unnamed: 0", axis = 1)["Comment"]
    y_train = y_train.drop("Unnamed: 0", axis = 1)["label"]
    y_test = y_test.drop("Unnamed: 0", axis = 1)["label"]


    Train = load_npz('ModelData/Train.npz')
    Test = load_npz('ModelData/Test.npz')

    # Starting with the Main App.

    # st.title('Youtube Universe of Comments.')\

    ml_model_list = ["SVM", "Linear Regression", "Decision Tree Classifier", "K-Nearest Neighbour Classifier", "Extra Tree Classifier",
                     "Random Forest Classifier", "XGBOOST Classifier", "Light Gradient Boosting Tree Classifier", "CATBOOST Classifier"]

    model_name = st.sidebar.selectbox("Select a ML model:", ml_model_list)

    if model_name == "SVM":
        C_select = st.select_slider("Select Value of C", [ 0.00001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
        gamma_select = st.selectbox("Choose Value of Gamma :", ['scale', 'auto'])
        degree_select = st.selectbox("Choose Value of Degree :", [2,3,4,5])
        kernel_select = st.selectbox("Choose the Kernel :", ["Linear Kernel", "RBF Kernel", "Polynomial Kernel"])
        kernel_dict = {"Linear Kernel" : "linear", "RBF Kernel" : "RBF", "Polynomial Kernel" : "poly"}
        model = svm_clf_poly = SVC(C = C_select, gamma = gamma_select, degree = degree_select, kernel = kernel_dict[kernel_select])
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, svm_clf_poly.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "Linear Regression":
        C_select = st.select_slider("Select Value of C", [ 0.0001, 0.001, 0.1, 0.25, 0.50, 0.75, 1, 1.25, 1.5, 1.75, 2.0])
        choose_max_iteration = st.slider("Select the value for the Maximum Iteration :", 100, 10000, 10)
        penalty_select = st.selectbox("Choose a Penalty :", ['l2', 'elasticnet'])
        model = LogisticRegression(C = C_select, max_iter=choose_max_iteration, penalty = penalty_select, random_state = 42)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "Decision Tree Classifier":
        criterion_select = st.selectbox('Choose the Criterion for the Model :', ['gini', 'entropy'])
        max_depth_select = st.selectbox("Select Value of max_depth :", [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 50, 75, 100])
        min_sample_split_select = st.selectbox("Select the value for the Minimum Sample Split :",  [2, 3, 4, 5, 6, 7, 8, 10])
        model = DecisionTreeClassifier(criterion = criterion_select, max_depth = max_depth_select, min_samples_split = min_sample_split_select)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "K-Nearest Neighbour Classifier":
        nearest_neighbour_select = st.select_slider('Choose the Number of Nearest Neighbours :', [3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 51])
        model = KNeighborsClassifier(n_neighbors = nearest_neighbour_select)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "Extra Tree Classifier":
        n_estimators_select = st.select_slider('Choose the Number of N-Estimators for the Model :', [ 50,100,250, 400, 600, 750, 1000])
        criterion_select = st.selectbox('Choose the Criterion for the Model :', ['gini', 'entropy'])
        max_depth_select = st.select_slider('Choose the maximum depth for the Model :', [5, 10, 20, 50, 100, 200, 300, 400, 500, 750])
        model = ExtraTreesClassifier(criterion = criterion_select, max_depth = max_depth_select, n_estimators = n_estimators_select, n_jobs=-1)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "Random Forest Classifier":
        n_estimators_select = st.select_slider('Choose the Number of N-Estimators for the Model :', [ 50,100,250, 400, 600, 750, 1000])
        max_depth_select = st.select_slider('Choose the maximum depth for the Model :', [5, 10, 20, 50, 100, 200, 300, 400, 500, 750])
        max_samples = st.select_slider('Choose the value for max. samples for the Model :', [0.6, 0.75, 1])
        model = RandomForestClassifier(n_estimators = n_estimators_select, max_depth = max_depth_select, max_samples = max_samples, n_jobs=-1)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "XGBOOST Classifier":
        learning_rate_select = st.select_slider('Choose the learning rate for the Model :', [0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 5])
        n_estimators_select = st.select_slider('Choose the Number of N-Estimators for the Model :', [100, 250, 500, 750, 1000])
        max_depth_select = st.select_slider('Choose the maximum depth for the Model :', [5, 10, 20, 50, 100, 200, 350, 500])
        colsample_bytree_select = st.select_slider('Choose the value for colsample bytree for the Model :', [0.6, 0.75, 1])
        subsample_select = st.select_slider('Choose the value for subsample bytree for the Model :', [0.6, 0.75, 1])
        model = XGBClassifier(n_estimators = n_estimators_select, colsample = colsample_bytree_select, subsample = subsample_select, depth = max_depth_select, learning_rate = learning_rate_select, iterations = 100, n_jobs=-1)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "Light Gradient Boosting Tree Classifier":
        max_depth_select = st.select_slider('Choose the maximum depth for the Model :', [5, 10, 20, 50, 100, 200, 350, 500])
        min_data_in_leaf_select = st.select_slider('Choose the Value for Min data in leaf :', [2,5,8,12,15,25,50])
        num_leaves_select = st.select_slider('Choose the Number of Leaves for the Model :', [20,50,100,250,500,750,1000,1500])
        model = LGBMClassifier(max_depth= max_depth_select, min_data_in_leaf = min_data_in_leaf_select, num_leaves = num_leaves_select, n_jobs=-1)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")

    if model_name == "CATBOOST Classifier":
        learning_rate_select = st.select_slider('Choose the learning rate for the Model :', [0.0001, 0.001, 0.01, 0.1, 1, 10])
        max_depth_select = st.select_slider('Choose the maximum depth for the Model :', [1, 2, 4, 8, 16])
        iteations_select = st.select_slider('Choose the Value for No. of iterations in the Model :', [10,20,50,100,200])
        model = CatBoostClassifier(depth= max_depth_select, learning_rate = learning_rate_select, iterations = iteations_select)
        model.fit(Train, y_train)
        plot_confusion_matrix(y_test, model.predict(Test))
        st.image("Confusion_Matrix_1.png")
        st.image("Confusion_Matrix_2.png")
        st.image("Confusion_Matrix_3.png")
    
    y_pred = model.predict(Test)
    f1_score_value = f1_score(y_test, y_pred, average='macro')
    text = "F1_Score of the Model is : " + str(round(f1_score_value, 3))
    st.header(text)

    return

main()