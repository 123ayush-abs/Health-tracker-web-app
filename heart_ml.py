import numpy as np 
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
heart_data=pd.read_csv('heart.csv')
#data cleaning 
def convert_sex(s):
    if(s=='M'):
        return 0
    else:
        return 1
def convert_exrecise(s):
    if(s=='N'):
        return 0
    else:
        return 1 
def convert_resting(s):
    if(s=='Normal'):
        return 0
    if(s=='ST'):
        return 1
    if(s=='LHV'):
        return 2
def convert_slope(s):
    if(s=='Up'):
        return 0
    else:
        return 1
def convert_chest(s):
    if(s=='ATA'):
        return 0
    if(s=='NAP'):
        return 1
    if(s=='ASY'):
        return 2
    else:
        return 3
    
#converting to numeric
heart_data['Sex']=heart_data['Sex'].apply(convert_sex)
heart_data['ST_Slope']=heart_data['ST_Slope'].apply(convert_slope)
heart_data['ExerciseAngina']=heart_data['ExerciseAngina'].apply(convert_exrecise)
heart_data['RestingECG']=heart_data['RestingECG'].apply(convert_resting)
heart_data['ChestPainType']=heart_data['ChestPainType'].apply(convert_chest)
heart_data=heart_data.fillna(0)
X=heart_data.drop(columns='HeartDisease',axis=1)
Y=heart_data['HeartDisease']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)
pickle.dump(model,open('heart1.sav','wb'))
# pred=model.predict(X_test)

# a_score=accuracy_score(Y_test,pred)  #checking accracy
# print(a_score)