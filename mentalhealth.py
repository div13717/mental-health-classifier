#Step 1 — using pandas,load dataset,explore data (shape, value_counts)remove unnamed column,remove null values,
#Step 2 — using nltk,remove special characters,lowercase everything,remove stopwords
#Step 3 — sklearn (TF-IDF),convert cleaned text to numbers
#Step 4 — sklearn (train_test_split), split data 80% train, 20% test
#Step 5 — sklearn (Logistic Regression) ,train model on 80% data
#Step 6 — sklearn (accuracy_score),test model on 20% data,get accuracy %
#Step 7 — pickle,save trained model to file
#Step 8 — vaderSentiment,add sentiment scores (positive/negative/neutral)
#Step 9 — streamlit,build web app ,deploy


import pandas as pd;
from colorama import Fore, Style

print("\n\n\n\n" + Style.BRIGHT + Fore.RED + "IMPORTING DATASET" + Style.RESET_ALL + "\n\n" );

data = pd.read_csv('mental_health.csv');
print(data) 

print("\n\n\n\n" + Style.BRIGHT + Fore.RED + "EXPLORATORY DATA ANALYSIS" + Style.RESET_ALL + "\n\n");
#column names in dataset
print("\n" + Style.BRIGHT + Fore.MAGENTA + "COLUMNS IN DATASET " + Style.RESET_ALL + "\n");
print(data.columns);
#data types in dataset
print("\n" +Style.BRIGHT + Fore.MAGENTA +"DATA TYPES IN DATASET"+ Style.RESET_ALL + "\n" );
print(data.dtypes);
#no of rows and columns in dataset
print("\n" + Style.BRIGHT + Fore.MAGENTA +"TOTAL NO OF ROWS AND COLUMNS IN DATASET"+ Style.RESET_ALL + "\n");
print(data.shape);
# Categories in status column
print("\n" + Style.BRIGHT + Fore.MAGENTA + "CATEGORIES IN STATUS COLUMN" + Style.RESET_ALL + "\n");
print(data['status'].value_counts());

print("\n\n\n\n" + Style.BRIGHT + Fore.RED +"CLEANING DATASET"+ Style.RESET_ALL + "\n\n");
#Removing the unnamed column from dataset
dataset= data.drop(columns = ['Unnamed: 0']);
print(dataset);
#null values in dataset
print("\n" + Style.BRIGHT + Fore.MAGENTA +"NULL VALUES IN DATASET"+ Style.RESET_ALL + "\n");
print(dataset.isnull());
#sum of null values in dataset
print("\n" + Style.BRIGHT + Fore.MAGENTA +"SUM OF NULL VALUES IN DATASET"+ Style.RESET_ALL + "\n");
print(dataset.isnull().sum());
# Removing null values from dataset
datasett = dataset.dropna()
print("\n" +Style.BRIGHT + Fore.MAGENTA + "TOTAL ROWS AFTER CLEANING" + Style.RESET_ALL + "\n")
print(datasett.shape);

print("\n" +Style.BRIGHT + Fore.RED + "CLEANING TEXT IN DATASET" + Style.RESET_ALL + "\n")


print("\n" +Style.BRIGHT + Fore.MAGENTA + "USING NLTK TO REMOVE STOP WORDS" + Style.RESET_ALL + "\n")

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

datasett['statement'] = datasett['statement'].apply(clean_text)

print(Style.BRIGHT + Fore.MAGENTA + "ROWS AFTER TEXT CLEANING: " + Style.RESET_ALL, len(datasett))
print(datasett[['statement', 'status']])



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

datasettt = datasett.dropna(subset=['statement', 'status'])
dataset = datasett.reset_index(drop=True)
X = datasett['statement']
y = datasett['status']
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)

print(Style.BRIGHT + Fore.MAGENTA + "\nSPLITTING DATA 80/20" + Style.RESET_ALL)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("TRAINING ROWS:", X_train.shape[0])
print("TESTING ROWS:", X_test.shape[0])

print(Style.BRIGHT + Fore.MAGENTA + "\nTRAINING MODEL..." + Style.RESET_ALL)
model = LogisticRegression(max_iter=1000, solver='saga')
model.fit(X_train, y_train)
print("TRAINING COMPLETE!")

print(Style.BRIGHT + Fore.MAGENTA + "\nTESTING ACCURACY" + Style.RESET_ALL)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(Style.BRIGHT + Fore.GREEN + f"ACCURACY: {accuracy * 100:.2f}%" + Style.RESET_ALL)

print(Style.BRIGHT + Fore.MAGENTA + "\nSAVING MODEL" + Style.RESET_ALL)
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
print(Style.BRIGHT + Fore.GREEN + "MODEL SAVED!" + Style.RESET_ALL)