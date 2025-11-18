import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Lets load all the instances required over here
with open('transformer.joblib','rb') as file:
    transformer = joblib.load(file)

with open('final_model.joblib','rb') as file:
    model = joblib.load(file)

st.title('INN HOTEL GROUP')
st.header(':blue[This application will predict the chances of booking cancellation]')

# Lets take input from users
amnth = st.slider('Select our month of arrival',min_value=1,max_value=12,step=1)
wkd_lambda = (lambda x:0 if x=='Mon' else 1 if x=='Tue' else
                 2 if x=='Wed' else 3 if x=='Thu' else
                 4 if x=='Fri' else 5 if x=='Sat' else 6)
awkd = wkd_lambda(st.selectbox('Select your weekday of arrival',['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))

dwkd = wkd_lambda(st.selectbox('Select your weekday of departure',['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))

wkend = st.number_input('Enter how many weekend nights you will stay',min_value=0)

wk = st.number_input('Enter how many week nights you will stay',min_value=0)

totn = wkend + wk

mrkt = (lambda x:0 if x=='Offline' else 1)(st.selectbox('How booking has been made',['Offline','Online']))

lt = st.number_input('How many days prior booking was made?',min_value=0)

price = st.number_input('How many days prior price per room?',min_value=0)

adults = st.number_input('How many adults members in your booking?',min_value=0)

spcl = st.selectbox('Select the specail request you made',[0,1,2,3,4,5])
park = (lambda x:0 if x=='No' else 1)(st.selectbox('Do you need parking space?',['No','Yes']))


# Transform the data
lt_t,price_t = transformer.transform([[lt,price]])[0]

# Create input list
input_list = [lt_t,spcl,price_t,adults,wkend,park,wk,mrkt,amnth,awkd,totn,dwkd]

# Predict the outcome

prediction = model.predict_proba([input_list])[:,1][0]

# Lets show the probability
if st.button('Predict'):
    st.success(f'The chances of booking cancellation is : {np.round(prediction*100,2)}%')
    if prediction>=0.3:
        st.warning('There are high chances of booking cancellation.Please reconfirm your bookingwith the hotel.')