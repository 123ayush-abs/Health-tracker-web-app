import pandas as pd
import numpy as np
import pickle
from PIL import Image
import streamlit as st 
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
a_sc1=0
st.set_page_config(page_title="Health Tracker",page_icon=" 🏬")
data_diab=pd.read_csv('diabetes.csv')
X=data_diab.drop(columns="Outcome",axis=1)
Y=data_diab["Outcome"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)

sres=""
sres1="" 
def predict_diab(inp_list):
    inp_arr=np.asarray(inp_list)
    resh=inp_arr.reshape(1,-1)
    pred=model.predict(resh)
    a_sc1=accuracy_score(Y_test, pred)*100
    if(pred[0]==1):
        sres1+='Yes'
        return "YOU HAVE DIABETES Symptoms"
    else:
        sres1+='No'
        return "NO DIABETES Symptoms"  
img=Image.open('healthcare (1).png')
st.image(img,width=100)
st.title("Welcome To Health Tracker Web App!!")
st.markdown("Predict The Chances of Your Health!!")
with st.sidebar:
    selected = option_menu("Main Menu", ["Diabetes Prediction",'Heart Disease Prediction'], 
        icons=["activity",'activity','file'], menu_icon="cast")


def Convert_chest(ChestPainType): 
    new_chest=""
    if(ChestPainType=='ATA'):
        new_chest=0
    elif(ChestPainType=='NAP'):
        new_chest=1
    elif(ChestPainType=='ASY'):
        new_chest=2
    else:
        new_chest=3
    return new_chest
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
def convert_exrecise(s):
    if(s=='N'):
        return 0
    else:
        return 1 

if(selected=='Diabetes Prediction'):
    Pregnancies=st.text_input("Enter Pregnancies Status(0 if Male)")
    Glucose=st.text_input("Enter Glucose Status")
    bp=st.text_input("Enter BloodPressure Status")
    skin=st.text_input("Enter SkinThickness Status")
    ins=st.text_input("Enter Insulin Status")
    bmi=st.text_input("Enter BMI Here")
    dpf=st.text_input("Enter DiabetesPedigreeFunction Status")
    age=st.text_input("Enter Your Age")
    if(st.button("Predict!!")):
        new_ll=[int(Pregnancies),int(Glucose),int(bp),int(skin),int(ins),float(bmi),float(dpf),int(age)]
        ress_ll=[]
        ress_ll.extend(new_ll)
        msg=predict_diab(ress_ll)
        st.success(msg)
elif(selected=="Heart Disease Prediction"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Age=st.text_input("Enter Your Age:")
        Sex=st.radio("Select Gender:",["M","F"])
        ChestPainType=st.selectbox("Select ChestPainType:",["ATA","NAP","ASY","TA"])
        RestingBP=st.text_input("Enter RestingBP:")
        Cholesterol=st.text_input("Enter Cholesterol level:")
        FastingBS=st.radio("Enter FastingBS:",[0,1])
        RestingECG=st.selectbox("Enter RestingECG:",["Normal","ST","LHV"])
        MaxHR=st.text_input("Enter MaxHR:")
        ExerciseAngina=st.radio("Enter ExerciseAngina:",["N","Y"])
        Oldpeak=st.text_input("Enter Oldpeak value:")
        ST_Slope=st.radio("Select ST_Slope:",["Up","Flat"])
        if(st.button("Predict!!")):
            heart_model=pickle.load(open('heart1.sav','rb'))
            new_sex= 0 if Sex=='M' else 1
            new_chest=Convert_chest(ChestPainType)
            new_resting_ecg=convert_resting(RestingECG)
            new_exercise=convert_exrecise(ExerciseAngina)
            new_slope=convert_slope(ST_Slope)
            new_list=[int(Age),new_sex,new_chest,int(RestingBP),int(Cholesterol),int(FastingBS),new_resting_ecg,int(MaxHR),new_exercise,float(Oldpeak),new_slope]
            new_input=[]
            new_input.extend(new_list)
            new_array=np.asarray(new_input)
            new_reshape=new_array.reshape(1,-1)
            res=heart_model.predict(new_reshape)
            if(res[0]==0):
                sres+='Yes'
                st.success("NO heart problem Symptoms"+"😊")
            else:
                sres+='No'
                st.warning(" Have Heart Problem Symptoms")
else:
    col11, col22= st.columns(2)
    col11.metric("Diabetes", str(a_sc1), "Yes")
    col22.metric("Heart Problems", "9 mph", "NO",delta_color="inverse")
   
                        
            





