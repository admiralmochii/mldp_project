import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor


print(sklearn.__version__)
st.write("""
# 🏠 Melbourne Housing Data Prediction App
This app predicts the price of a property in Melbourne based on a variety of factors!
""")
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")

st.subheader("Parameters")

# property count put to max 21650
def user_input_features():
    rooms = st.slider('No. of bedrooms', 1.0, 3.0, 8.0, step=1.0)
    date = st.slider('Date of sale (YYYYMM)', max_value=datetime(2017,12,31),min_value=datetime(2016,1,1), format="YYYYMM")
    distance = st.slider('Distance from Melbourne CBD (km)', 0.0, 9.75, 29.8)
    bathrooms = st.slider('No. of Bathrooms', 0.0, 1.0, 4.0, step=1.0)
    cars = st.slider('No. of carspots', 0.0, 2.0, 10.0, step=1.0)
    land = st.slider('Land Size (meter square)', 0.0, 451.0, 4497.0)
    prop_count = st.slider('No. of properties in the area', 389.0, 7416.0, 21650.0)
    house_type = st.selectbox("Type of property", options=(
        "Bedroom(s)",
        "House, Cottage, Villa, Semi, Terrace",
        "Unit, Duplex",
        "Townhouse",
        "Developmental Site",
        "Other Residential"
    ))  

    if house_type == "Bedroom(s)":
        house_type = "br"
    if house_type == "House, Cottage, Villa, Semi, Terrace":
        house_type = "h"
    if house_type == "Unit, Duplex":
        house_type = "u"
    if house_type == "Townhouse":
        house_type = "t"
    if house_type == "Developmental Site":
        house_type = "dev site"
    if house_type == "Other Residential":
        house_type = "o res"
    
    seller = st.selectbox("Property Agent", options=(
        "Barry","Biggin","Brad","Buxton","Fletchers","Gary","Greg","Harcourts","Hodges","Jas","Jellis","Kay","Love","Marshall","McGrath",
        "Miles","Nelson","Noel","RT","Ray","Stockdale","Sweeney","Village","Williams","Woodards","YPA","hockingstuart", "Others"
    ))

    region = st.selectbox("General Region", options=(
        "Eastern Metropolitan","Eastern Victoria","Northern Metropolitan","Northern Victoria","South-Eastern Metropolitan",
        "Southern Metropolitan","Western Metropolitan","Western Victoria"
    ))

    council_area = st.selectbox("Council Area", options=(
        "Banyule","Bayside","Boroondara","Brimbank","Casey","Darebin","Glen Eira","Greater Dandenong","Hobsons Bay","Hume","Kingston",
        "Knox","Manningham","Maribyrnong","Maroondah","Melbourne","Melton","Monash","Moonee Valley","Moreland","Nillumbik","Port Phillip",
        "Stonnington","Whitehorse","Whittlesea","Wyndham","Yarra","Yarra Ranges","Unavailable","None"
    ))

    suburb = st.selectbox("Suburb", options=(
        "Ascot Vale","Balwyn","Balwyn North","Bentleigh","Bentleigh East","Brighton","Brighton East","Brunswick","Brunswick West","Camberwell",
        "Carnegie","Coburg","Doncaster","Elwood","Essendon","Fawkner","Footscray","Glen Iris","Glenroy","Hampton","Hawthorn","Hawthorn East",
        "Keilor East","Kensington","Kew","Malvern East","Moonee Ponds","Newport","Northcote","Pascoe Vale", "Port Melbourne","Prahran", "Preston",
        "Reservoir","Richmond","South Yarra","St Kilda","Sunshine","Surrey Hills","Thornbury","West Footscray","Williamstown","Yarraville"
    ))

    date_str = date.strftime('%Y%m') 

    data = {'Rooms': rooms,
            'Date': date_str,
            'Distance': distance,
            'Bathroom': bathrooms,
            'Car': cars,
            'Landsize': land,
            'Propertycount': prop_count,
            'Type_' + house_type: 1 ,
            "SellerG_" + seller: 1,
            "Regionname_" + region: 1,
            "CouncilArea_" + council_area: 1,
            "Suburb_" + suburb: 1 
            }
    
    features = pd.DataFrame(data, index=[0])
    
    for feature in expected_features:
        if feature not in features.columns:
            features[feature]=0.0
    return features[expected_features]





mldp_gbr = pickle.load(open("model_project_gbr.pkl","rb"))
expected_features = pickle.load(open("mldp_project_features.pkl","rb"))

df = user_input_features()

st.write(df)
predicted_price = mldp_gbr.predict(df)


st.sidebar.subheader('Predicted price of the property:')

st.sidebar.write("# A$ " + str(round(predicted_price[0], 2)))