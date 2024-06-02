import streamlit as st
import numpy as np
import pickle

class_pickle=open('neural_network.pickle','rb')
model=pickle.load(class_pickle)
class_pickle.close()
st.title('Mobile Price Classification App')
st.subheader('We use the Mobile price dataset from kaggle to predict the category of the cellphone based on its properties ')
battery=st.number_input('battery_power (mAh)',min_value=0)
blue=st.number_input('bluetooth or not(binary)',min_value=0,max_value=1)
clock=st.number_input('clock_speed (MHz)',min_value=0)
sim=st.number_input('dual_sim or not (binary)',min_value=0,max_value=1)
front_cam=st.number_input('front_camera_pixels (MegaBytes)',min_value=0)
four_g=st.number_input('4G or not (binary)',min_value=0,max_value=1)
inter_memory=st.number_input('internal_memory(GigaBytes)',min_value=0)
mobile_depth=st.number_input('mobile_depth(Cm)',min_value=0)
weight=st.number_input('weight (G)',min_value=0)
n_cores=st.number_input('number_of_cores',min_value=1)
primary_camera=st.number_input('primary_camera_pixels(MegaBytes)',min_value=0)
px_height=st.number_input('pixel_resolution_height',min_value=0)
px_width=st.number_input('pixel_resolution_width',min_value=0)
ram=st.number_input('Ram',min_value=0)
sc_h=st.number_input('screen_Hight(Cm)',min_value=0)
sc_w=st.number_input('screen_width(Cm)',min_value=0)
talk_time=st.number_input('longest time that a single battery charge will last when you are (H)',min_value=0)
three_g=st.number_input('3g or not (binary)',min_value=0,max_value=1)
touch_screen=st.number_input('touchscreen or Not',min_value=0,max_value=1)
wifi=st.number_input('has wifi or not (binary)',min_value=0,max_value=1)
new_prediction=model.predict(np.array([[battery,blue,clock,sim,front_cam,four_g,inter_memory,mobile_depth,weight,n_cores,primary_camera
,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi]]))
categories=['Low','Medium','High','VeryHigh']
st.write(f'Based on features selected , your phone category is {categories[np.argmax(new_prediction)]} ')
