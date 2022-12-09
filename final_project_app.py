import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv", na_values=" ")
def clean_sm(x):
    x = np.where(x==1, 1, 0)
    return x
ss = s[['income','educ2','par','marital','gender','age', 'web1h']]
ss = ss.assign(sm_li = clean_sm(ss["web1h"]))
ss = ss.drop(['web1h'], axis=1)
ss = ss[ss['income'] <= 9]
ss = ss[ss['educ2'] <= 8]
ss = ss[ss['par'] <= 2]
ss["par"] = np.where(ss["par"] == 2, 0, 1)
ss = ss[ss['marital'] <= 6]
ss["marital"] = np.where(ss["marital"] >= 2, 0, 1)
ss = ss[ss['gender'] <= 2]
ss['gender'] = np.where(ss["gender"] == 2, 1, 0)
ss = ss[ss['age'] <= 98]
ss = ss.rename(columns = {"educ2": "education", "marital":"married", "par": "parent", "gender": "female"})

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 
                                                
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


st.markdown("# Predicting LinkedIn Users with Machine Learning ")

st.markdown("#### By: Chandini Ramesh ")

st.markdown("# Details of Potential LinkedIn User ")

income = st.selectbox(label="What income level does this person fall under? (in USD)",
options=("Less than $10k", " $10k-$20k", " $20k-$30k", " $30k-$40k", " $40k-$50k", " $50k-$75k", " $75k-$100k", " $100k-$150k", "More than $150K"))

if income == "Less than $10k":
    income = 1
elif income == "$10k-$20k":
    income = 2
elif income == "$20k-$30k":
    income = 3
elif income == "$30k-$40k":
    income = 4
elif income == "$40k-$50k":
    income = 5
elif income == "$50k-$75k":
    income = 6
elif income == "$75k-$100k":
    income = 7
elif income == "$100k-$150k":
    income = 8
else:
    income = 9

educ = st.selectbox(label="What education level does this person fall under?",
options=("Less than High School", "High school - No Diploma", "High School Graduate", "Some College, No Degree", "Two-Year Associate Degree", "Bachelor's Degree", "Some Postgraduate Schooling, No Degree", "Master's Degree or More"))

if educ == "Less than High School":
    educ = 1
elif educ == "High school - No Diploma":
    educ = 2
elif educ == "High School Graduate":
    educ = 3
elif educ == "Some College, No Degree":
    educ = 4
elif educ == "Two-Year Associate Degree":
    educ = 5
elif educ == "Bachelor's Degree":
    educ = 6
elif educ == "Some Postgraduate Schooling, No Degree":
    educ = 7
else:
    educ = 8

parent = st.selectbox(label="Is this person a parent?",
options=("Yes", "No"))

if parent == "Yes":
    parent = 1
else:
    parent = 0

married = st.selectbox(label="Is this person married?",
options=("Yes", "No"))

if married == "Yes":
    married = 1
else:
    married = 0

female = st.selectbox(label="Does this identify as male or female?",
options=("Female", "Male"))

if female == "Female":
    female = 1
else:
    female = 0

age = st.slider(label="How old is this person?", 
          min_value=1,
         max_value=98,
          value=25)

person = [income, educ, parent, married, female, age]

final_prediction = lr.predict([person])
if final_prediction == 1:
    final_prediction1 = "This person is a LinkedIn User"
else: 
    final_prediction1 = "This person is not a LinkedIn User"


st.write(f"# {final_prediction1}")


