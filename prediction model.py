#!/usr/bin/env python
# coding: utf-8

# **Early stage Diabetes Risk prediction**

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
dataset = pd.read_csv('C:/Users/ZhuanZ/Documents/diabitices/Project4-diabetes_data.csv')
dataset.head()

# Checking whether dataset has null values
# sns.heatmap(dataset.isnull())
# plt.show()





dataset['class'].value_counts()

# Mapping text into values
dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':0})
dataset['class'] = dataset['class'].map({'Positive':1,'Negative':0})
dataset['Polyuria'] = dataset['Polyuria'].map({'Yes':1,'No':0})
dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes':1,'No':0})
dataset['sudden weight loss'] = dataset['sudden weight loss'].map({'Yes':1,'No':0})
dataset['weakness'] = dataset['weakness'].map({'Yes':1,'No':0})
dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes':1,'No':0})
dataset['Genital thrush'] = dataset['Genital thrush'].map({'Yes':1,'No':0})
dataset['visual blurring'] = dataset['visual blurring'].map({'Yes':1,'No':0})
dataset['Itching'] = dataset['Itching'].map({'Yes':1,'No':0})
dataset['Irritability'] = dataset['Irritability'].map({'Yes':1,'No':0})
dataset['delayed healing'] = dataset['delayed healing'].map({'Yes':1,'No':0})
dataset['partial paresis'] = dataset['partial paresis'].map({'Yes':1,'No':0})
dataset['muscle stiffness'] = dataset['muscle stiffness'].map({'Yes':1,'No':0})

dataset['Alopecia'] = dataset['Alopecia'].map({'Yes':1,'No':0})
dataset['Obesity'] = dataset['Obesity'].map({'Yes':1,'No':0})

# EDA
# Analysing the correlation between independent and dependent variables
corrdata = dataset.corr()

# sns.histplot(dataset['Age'], bins=30)
# plt.show()





# Age/class(dependent variable)
# sns.barplot(x='class', y='Age', data=dataset)
# plt.show()





# ds = dataset['class'].value_counts().reset_index()
# ds.columns = ['class', 'count']
# plot = ds.plot.pie(y='count')
# plt.show()





# Gender
# sns.countplot(x='class', data=dataset, hue='Gender')
# plt.show()





# Polyuria
# sns.catplot(x="Polyuria", y="class", kind="point", data=dataset)
# plt.show()





# Polydipsia
# sns.barplot(x='Polydipsia', y='class', data=dataset)
# plt.show()





# Sudden weight loss
# sns.countplot(x='class', data=dataset, hue='sudden weight loss')
# plt.show()





# Polyphagia
# sns.countplot(x='class', data=dataset, hue='Polyphagia')
# plt.show()





# sns.barplot(x='Polyphagia', y='class', data=dataset)
# plt.show()





# Genital Thrush
# sns.barplot(x='class', y='Genital thrush', data=dataset)
# plt.show()





# Partial paresis
# sns.barplot(x='class', y='partial paresis', data=dataset)
# plt.show()





# Alopecia
# sns.barplot(x='Alopecia', y='class', data=dataset)
# plt.show()






# Visual blurring
# sns.barplot(x="visual blurring", y="class", data=dataset)
# plt.show()





# Itching
# sns.barplot(x="Itching", y="class", data=dataset)
# plt.show()





# Obesity
# sns.barplot(x='Obesity', y='class', data=dataset)
# plt.show()





# Irritability
# sns.barplot(x='Irritability', y='class', data=dataset)
# plt.show()



X1 = dataset.iloc[:,0:-1]
y1 = dataset.iloc[:,-1]

X1.columns

# Feature selection using selectkbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
best_feature = SelectKBest(score_func=chi2, k=10)
fit = best_feature.fit(X1, y1)

dataset_scores = pd.DataFrame(fit.scores_)
dataset_cols = pd.DataFrame(X1.columns)

featurescores = pd.concat([dataset_cols, dataset_scores], axis=1)
featurescores.columns = ['column', 'scores']

# These are the variables with their feature scores, their importance/contribution towards class
featurescores

# Top 10 features
print(featurescores.nlargest(10, 'scores'))

featureview = pd.Series(fit.scores_, index=X1.columns)
# featureview.plot(kind='barh')
# plt.show()

input("Press Enter to continue...")

# Checking the variance of each feature
from sklearn.feature_selection import VarianceThreshold
feature_high_variance = VarianceThreshold(threshold=(0.5*(1-0.5)))
falls = feature_high_variance.fit(X1)

dataset_scores1 = pd.DataFrame(falls.variances_)
dat1 = pd.DataFrame(X1.columns)

high_variance = pd.concat([dataset_scores1, dat1], axis=1)
high_variance.columns = ['variance', 'cols']

high_variance[high_variance['variance'] > 0.2]

X = dataset[['Polydipsia', 'sudden weight loss', 'partial paresis', 'Irritability', 'Polyphagia', 'Age', 'visual blurring']]
y = dataset['class']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardization of independent variables
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)

# Cross validation test for training data
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=lg, X=X_train, y=y_train, cv=10)
print("accuracy is {:.2f} %".format(accuracies.mean()*100))
print("standard deviation is {:.2f} %".format(accuracies.std()*100))

# Prediction
pre = lg.predict(X_test)

# Confusion matrix
logistic_regression = accuracy_score(pre, y_test)
print("accuracy score:",)
print(accuracy_score(pre, y_test))
print("Confusion Matrix:",)
print(confusion_matrix(pre, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre, y_test))

# SVM
from sklearn.svm import SVC
sv = SVC(kernel='linear', random_state=0)
sv.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=sv, X=X_train, y=y_train, cv=10)
print("mean accuracy is {:.2f} %".format(accuracies.mean()*100))
print("standard deviation is {:.2f} %".format(accuracies.std()*100))

pre1 = sv.predict(X_test)

svm_linear = accuracy_score(pre1, y_test)
print("accuracy score:",)
print(accuracy_score(pre1, y_test))
print("Confusion Matrix:",)
print(confusion_matrix(pre1, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre1, y_test))

from sklearn.svm import SVC
svrf = SVC(kernel='rbf', random_state=0)
svrf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svrf, X=X_train, y=y_train, cv=10)
print("mean accuracy is {:.2f} %".format(accuracies.mean()*100))
print("standard deviation is {:.2f} %".format(accuracies.std()*100))

pre2 = svrf.predict(X_test)

svm_rbf = accuracy_score(pre2, y_test)
print(accuracy_score(pre2, y_test))
print(confusion_matrix(pre2, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre2, y_test))

# KNN
from sklearn.neighbors import KNeighborsClassifier
score = []

for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    pre3 = knn.predict(X_test)
    ans = accuracy_score(y_test, pre3)  # Ensure y_test is the true labels
    print("Accuracy score {}: {}".format(i, round(100 * ans, 2)))
    score.append(round(100 * ans, 2))

print("Top 5 accuracy scores:", sorted(score, reverse=True)[:5])

best_score = sorted(score, reverse=True)[:1]
print("Best accuracy score:", best_score)

# Naive bayes-Gaussian NB
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=gb, X=X_train, y=y_train, cv=10)
print("Mean accuracy is {:.2f} %".format(accuracies.mean()*100))
print("standard Deviation is {:.2f} %".format(accuracies.std()*100))

pre4 = gb.predict(X_test)

Naive_bayes_Gaussian_nb = accuracy_score(pre4, y_test)
print("accuracy score:",)
print(accuracy_score(pre4, y_test))
print("Confusion Matrix:",)
print(confusion_matrix(pre4, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre4, y_test))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(criterion='gini')
dc.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=dc, X=X_train, y=y_train, cv=10)
print("accuracy is {:.2f} %".format(accuracies.mean()*100))
print("std is {:.2f} %".format(accuracies.std()*100))

pre5 = dc.predict(X_test)

Decisiontress_classifier = accuracy_score(pre5, y_test)
print("accuracy score:",)
print(accuracy_score(pre5, y_test))
print("Confusion Matrix:",)
print(confusion_matrix(pre5, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre5, y_test))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
estime = []
for i in range(1, 100):
    rc = RandomForestClassifier(n_estimators=i, criterion='entropy', random_state=0)
    rc.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=rc, X=X_train, y=y_train, cv=10)
print("accuracy is {:.2f} %".format(accuracies.mean()*100))
print("std is {:.2f} %".format(accuracies.std()*100))

pre6 = rc.predict(X_test)

Random_forest = accuracy_score(pre6, y_test)
print("accuracy score:",)
print(accuracy_score(pre6, y_test))
print("Confusion Matrix:",)
print(confusion_matrix(pre6, y_test))

from sklearn.metrics import classification_report
print(classification_report(pre6, y_test))

# Accuracies of all classification model overview
print('Logistic regression:', logistic_regression)
print('svmlinear:', svm_linear)
print('svmrbf:', svm_rbf)
print('knn:', knn)
print('naive bayes:', Naive_bayes_Gaussian_nb)
print('Decision tree:', Decisiontress_classifier)
print('Random forest:', Random_forest)

# The best model is SVM, KNN and Random forest with 98% Accuracy

def choose_model():
    """Function to choose a model for prediction."""
    print("Available models:")
    print("1. Logistic Regression")
    print("2. SVM (Linear Kernel)")
    print("3. SVM (RBF Kernel)")
    print("4. KNN")
    print("5. Naive Bayes")
    print("6. Decision Tree")
    print("7. Random Forest")
    
    choice = int(input("Select a model (1-7): "))
    models = {
        1: lg,
        2: sv,
        3: svrf,
        4: knn,
        5: gb,
        6: dc,
        7: rc
    }
    
    return models.get(choice, None)

def get_user_input():
    """Function to get user input for prediction"""
    print("Please enter the following details:")
    Polydipsia = int(input("Polydipsia (1 for Yes, 0 for No): "))
    sudden_weight_loss = int(input("Sudden weight loss (1 for Yes, 0 for No): "))
    partial_paresis = int(input("Partial paresis (1 for Yes, 0 for No): "))
    Irritability = int(input("Irritability (1 for Yes, 0 for No): "))
    Polyphagia = int(input("Polyphagia (1 for Yes, 0 for No): "))
    Age = int(input("Age (in years): "))
    visual_blurring = int(input("Visual blurring (1 for Yes, 0 for No): "))
    
    # Return input as a numpy array for prediction
    return np.array([[Polydipsia, sudden_weight_loss, partial_paresis, 
                    Irritability, Polyphagia, Age, visual_blurring]])

# Choose model and get user input
selected_model = choose_model()
if selected_model is None:
    print("Invalid model selection! Exiting...")
    exit()

user_input = get_user_input()
user_input_scaled = ss.transform(user_input)  # Scale the input
prediction = selected_model.predict(user_input_scaled)

# Display result
if prediction[0] == 1:
    print("\nPrediction: Positive for diabetes risk")
else:
    print("\nPrediction: Negative for diabetes risk")
