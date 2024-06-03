import streamlit as st
import pandas as pd
import numpy as np
import joblib             

 
from model import ordinal_encoder,get_prediction
from sklearn.ensemble import ExtraTreesClassifier


st.set_page_config(page_title ="Accident Severity Prediction App", page_icon="🚧", layout="wide")




options_light_condition = ['Darkness - lights lit' ,'Darkness - lights unlit','Darkness - no lighting', 'Daylight']
options_road_surface =['Dry','Flood over 3cm. deep', 'Snow', 'Wet or damp']
options_junction_type=['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'X Shape','Y Shape' ]
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence']




features=['light_condition', 'casualties','vehicles_involved','driver_Age','minute','day_of_week','driving_experience','road_surface_conditions','junction_type','hour']
 

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)

st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"]
        {
            background-image: url('https://t4.ftcdn.net/jpg/06/89/58/59/240_F_689585984_8kU3bCXzIpOQucVToyp5sJb9ZDYB29o2.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def main():
    st.markdown("""
    <style>
    .dark-subheader {
        color: black;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.form('prediction form'):
        st.subheader('Enter the input for the following features:')
        light_condition = st.selectbox("Select Light Condition: ", options=options_light_condition)
        casualties = st.slider("Casualties: ", 1, 8, value=0, format="%d")
        vehicles_involved = st.slider("Number of Vehicles Involved: ", 1, 7, value=0, format="%d")
        driver_Age = st.selectbox("Select Driver Age: ", options=options_age)
        minute = st.slider("At what Minute: ",0,59 , value=0, format="%d")
        day_of_week = st.selectbox("Select Day of week:", options =options_day)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        road_surface_conditions =st.selectbox("Select Road Surface Condition: ", options=options_road_surface)
        junction_type=st.selectbox("Select Junction Type: ", options=options_junction_type)
        hour = st.slider("Pickup hour:",0,23,value=0, format="%d")
        
        
        
        submit = st.form_submit_button('Predict')
        
    if submit:
        light_condition=ordinal_encoder(light_condition,options_light_condition)
        driver_Age = ordinal_encoder(driver_Age,options_age)
        day_of_week = ordinal_encoder(day_of_week,options_day)
        driving_experience = ordinal_encoder(driving_experience,options_driver_exp)
        road_surface_conditions = ordinal_encoder( road_surface_conditions,options_road_surface)
        junction_type=ordinal_encoder(junction_type,options_junction_type)
        
        
        
        data  = np.array([light_condition, casualties,vehicles_involved,driver_Age,minute,day_of_week,driving_experience,road_surface_conditions,junction_type,hour]).reshape(1,-1)
        
        
        pred  = get_prediction(data=data)
        st.balloons()
        st.write(f"The predicted severity is: {pred[0]}")
        
    
        
if __name__ == '__main__':
    main()
        
        
        
        
 