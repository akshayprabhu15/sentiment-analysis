import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer


df = pd.read_excel(r"C:\Users\Ak\Desktop\sentiment_new\Comments_4_22_Survey_label_manual.xlsx")

df_new = df[['Location','Business Unit','Question','Answer','Label_Data']]

df_new.dropna(subset=['Answer'],inplace=True)
df_new=df_new.reset_index(drop=True)

spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]


#text preprocess
stop = stopwords.words('english')  
df_new['Answer_processed'] = df_new['Answer'].apply(lambda j: ' '.join([item for item in j.split() if item not in stop]))

stemmer = PorterStemmer()
df_new['Answer_processed'] = df_new['Answer_processed'].apply(lambda j: ' '.join([stemmer.stem(item) for item in j.split()]))

for char in spec_chars:
    df_new['Answer_processed'] = df_new['Answer_processed'].str.replace(char, '').str.lower()
    

#drop rows with number of words less than length=2
df_new=df_new[~df_new['Answer_processed'].str.split().str.len().lt(2)] 
df_new=df_new.reset_index(drop=True)


#shuffle
from sklearn.utils import shuffle
df_new = shuffle(df_new)

#train and test
X = df_new[['Answer_processed']]
y = df_new[["Label_Data"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#vectorize
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000,binary=True)

x_train_vect = vect.fit_transform(x_train.Answer_processed)


#naive bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train_vect, y_train)

x_test_vect = vect.transform(x_test.Answer_processed)
y_pred = nb.predict(x_test_vect)

#accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))




#applying the model on test data
df_test = pd.read_excel(r"C:\Users\Ak\Desktop\sentiment_new\Mock_Results_Text_label_manual.xlsx")
df_test_new = df_test[['Location','Business Unit','Question','Answer','Label_Data']]

df_test_new.dropna(subset=['Answer'],inplace=True)
df_test_new=df_test_new.reset_index(drop=True)


#preprocess test data
stop = stopwords.words('english')  
df_test_new['Answer_processed'] = df_new['Answer'].apply(lambda j: ' '.join([item for item in j.split() if item not in stop]))

stemmer = PorterStemmer()
df_test_new['Answer_processed'] = df_test_new['Answer_processed'].apply(lambda j: ' '.join([stemmer.stem(item) for item in j.split()]))

for char in spec_chars:
    df_test_new['Answer_processed'] = df_test_new['Answer_processed'].str.replace(char, '').str.lower()
    

#drop rows with number of words less than length=2
df_new=df_new[~df_new['Answer_processed'].str.split().str.len().lt(2)] 
df_new=df_new.reset_index(drop=True)

#apply model
df_test_predict = df_test_new[["Answer_processed"]]
df_test_vect = vect.transform(df_test_predict.Answer_processed)
test_model = nb.predict(df_test_vect)

df_test_new['Predicted_Label_Data']=test_model
df_test_new.to_excel(r'C:\Users\Ak\Desktop\sentiment_new\Mock_Results_Text_Label_Predictions.xlsx',index=False)

print("Accuracy: {:.2f}%".format(accuracy_score(df_test_new.Label_Data, test_model) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(df_test_new.Label_Data, test_model))


#join two datasets and apply model
df_1 = df_new[['Location','Business Unit','Question',"Answer","Answer_processed","Label_Data"]]
df_2 = df_test_new[['Location','Business Unit','Question',"Answer","Answer_processed","Label_Data"]]

df_concat = pd.concat([df_1,df_2], ignore_index=True)


df_test_concat = df_concat[["Answer_processed"]]
df_concat_vect = vect.transform(df_test_concat.Answer_processed)
test_concat_model = nb.predict(df_concat_vect)

df_concat['Predicted_Label_Data']=test_concat_model

print("Accuracy: {:.2f}%".format(accuracy_score(df_concat.Label_Data, test_concat_model) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(df_concat.Label_Data, test_concat_model))


df_concat.to_excel(r'C:\Users\Ak\Desktop\sentiment_new\Comment_Survey_Mock_Results_Concat_Predictions.xlsx',index=False)



