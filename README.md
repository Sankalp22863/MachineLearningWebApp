# Youtube Universe of Comments.

## A project to clean up the youtube comment section using the AI/ML algorithms.

## A part of the Machine Learning project.

### Table of Contents

   + [Installation](#installation)
   + [Repository Structure](#repository-structure)
   + [Project Motivation](#project-motivation)
   + [File Descriptions](#file-descriptions)
   + [Models Used](#models-used)
   + [Instructions To Run](#instructions-to-run)
   + [Acknowledgement, Author and Licensing](#acknowledgement--author-and-licensing)

### Installation
The code should run with no issues using Python versions 3.* Using Jupyter notebook from Anaconda is recommended. You may use other data visualization tools like Tableau for reference. The libraries required with appropriate versions can be found in [requirements.txt](https://github.com/Sankalp22863/MachineLearningWebApp/blob/master/requirements.txt).

### Repository structure



### Project Motivation



### File Descriptions
[data](https://github.com/Sankalp22863/MachineLearningWebApp/blob/master/Data.csv) - This data file, attached to the repository contains all the data. It contains different kinds of comments, classified into 3 categories - Non-offensive, Hate-Speech and Abusive. The data has been collected using Youtube API scraping. The categories ave been assigned manually.

[data cleaning/preprocessing](https://github.com/Sankalp22863/MachineLearningWebApp/blob/master/DataCleaning.py) 

[Youtube API- Scraping Comments](https://github.com/Sankalp22863/MachineLearningWebApp/blob/master/YoutubeAPI.py)

[Models](https://github.com/Sankalp22863/MachineLearningWebApp/blob/master/model.py)


### Models Used
* Logistic regression:
* Support Vector Machine:
* Support Vector Machine with Linear Kernel:
* Support Vector Machine using RBF Kernel:
* Support Vector Machine using Polynomial Kernel:
* Decision Tree Classifier:
* K-Nearest Neighbour Classifier:
* Extra Tree Classifier:
* Random Forest Classifier:
* Model Parameter Optimization using GridSearchCV:

### Instructions To Run
First install the dependencies
```bash
pip install -r requirements.txt
```

Now to run the code on streamlit
```bash
streamlit run main.py
```


### Acknowledgement, Author and Licensing
For the project, I give credit to 
* Dr. Ankit Bhurane for guiding us in this project
* Dr. Andrew Ng for his insightful course on Coursera

The code can be freely used by any individual or organization for their needs. 
MIT LICENSED.
