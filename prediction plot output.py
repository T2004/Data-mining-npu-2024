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
sns.heatmap(dataset.isnull())
plt.show()




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

sns.histplot(dataset['Age'], bins=30)
plt.show()




# Age/class(dependent variable)
sns.barplot(x='class', y='Age', data=dataset)
plt.show()




ds = dataset['class'].value_counts().reset_index()
ds.columns = ['class', 'count']
plot = ds.plot.pie(y='count')
plt.show()




# Gender
sns.countplot(x='class', data=dataset, hue='Gender')
plt.show()




# Polyuria
sns.catplot(x="Polyuria", y="class", kind="point", data=dataset)
plt.show()




# Polydipsia
sns.barplot(x='Polydipsia', y='class', data=dataset)
plt.show()




# Sudden weight loss
sns.countplot(x='class', data=dataset, hue='sudden weight loss')
plt.show()




# Polyphagia
sns.countplot(x='class', data=dataset, hue='Polyphagia')
plt.show()




sns.barplot(x='Polyphagia', y='class', data=dataset)
plt.show()




# Genital Thrush
sns.barplot(x='class', y='Genital thrush', data=dataset)
plt.show()




# Partial paresis
sns.barplot(x='class', y='partial paresis', data=dataset)
plt.show()




# Alopecia
sns.barplot(x='Alopecia', y='class', data=dataset)
plt.show()





# Visual blurring
sns.barplot(x="visual blurring", y="class", data=dataset)
plt.show()




# Itching
sns.barplot(x="Itching", y="class", data=dataset)
plt.show()




# Obesity
sns.barplot(x='Obesity', y='class', data=dataset)
plt.show()




# Irritability
sns.barplot(x='Irritability', y='class', data=dataset)
plt.show()
