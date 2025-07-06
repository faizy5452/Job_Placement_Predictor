import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

lg=pickle.load(open('placement.pkl','rb'))

image=Image.open('Job-Placement-Agency.jpg')
st.image(image,use_container_width=True)
st.title('Job Placement Prediction App')

input_text=st.text_input('Enter All Features :')

if input_text:
    input_list=input_text.split(',')
    np_dp=np.asarray(input_list,dtype=float)
    prediction=lg.predict(np_dp.reshape(1,-1))

    if prediction[0]==1:
        st.write('Job placement is successful!')
    st.write('Job placement is unsuccessful!')