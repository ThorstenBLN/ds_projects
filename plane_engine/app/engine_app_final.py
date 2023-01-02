import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
from csv import reader
from playsound import playsound  


# set the wide format for streamlit
st.set_page_config(layout="wide")

# [data-testid="stAppViewContainer"]
def add_bg_from_url():
    '''downloads the image from unsplash and sets it as background for the webpage'''
    page_bg_img = """
        <style>
        .stApp {
            background-image: url(https://images.unsplash.com/flagged/photo-1579750481098-8b3a62c9b85d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1460&q=80);
            background-size: cover;
        }
        </style>
        """
    st.markdown(page_bg_img,
         unsafe_allow_html=True
     )

# add the background image
add_bg_from_url() 

# used features for data and plots
FEATURES = ['setting1', 'setting2', 's2', 's3', 's4',
       's6', 's9', 's10', 's11', 's12', 's13', 's15', 's17', 's20', 's21',
       'mean_setting1', 'mean_setting2', 'mean_s2', 'mean_s3', 'mean_s4',
       'mean_s6', 'mean_s9', 'mean_s10', 'mean_s11', 'mean_s12', 'mean_s13',
       'mean_s15', 'mean_s17', 'mean_s20', 'mean_s21', 'var_setting1',
       'var_setting2', 'var_s2', 'var_s3', 'var_s4', 'var_s6', 'var_s9',
       'var_s10', 'var_s11', 'var_s12', 'var_s13', 'var_s15', 'var_s17',
       'var_s20', 'var_s21']

# dictionary containing the names of the features
feature_dict= {'setting1':'Setting1', 'setting2':'Setting2',
                's1':'Tot. temp. fan inlet','s2':'Tot. temp. LPC outlet',
                's3':'Tot. temp. HPC outlet','s4':'Tot. temp. LPT outlet',
                's5':'Pressure fan inlet','s6':'Tot. press. bypass-duct',
                's7':'Tot. press. HPC outlet','s8':'Phys. fan speed',
                's9':'Phys. core speed','s10':'Engine press. ratio',
                's11':'Stat. Press. HPC outlet','s12':'Ratio fuel flow to Ps30',
                's13':'Corrected fan speed','s14':'Corrected Core speed',
                's15':'Bypass ratio','s16':'Burner fuel-air ratio','s17':'Bleed enthalpy',
                's18':'demanded fan speed','s19':'demanded corr. fan speed',
                's20':'HPT coolant bleed','s21':'LPT coolant bleed',
}
# defining the colors of the simulation
MAIN_COLOR = '#FF8888'
BG_PAPER_COLOR = 'rgba(33,17,5,0.6)'
BG_PLOT_COLOR = 'rgba(215,215,215,1)'

NO_SHOWN_FEATURES = 15
UNSHOWN_FEATURES = FEATURES[NO_SHOWN_FEATURES:]

# get the data only once
@st.cache
def get_data():
    '''load the data from the prepared np.array dictionary from the classification 
    and the regression model'''
    data_class = np.load('../simulation_data/data_class.npy', allow_pickle=True)
    data_reg = np.load('../simulation_data/data_regr.npy', allow_pickle=True)
    # load the ids for the engines
    file_id = '../simulation_data/ids.txt'
    with open(file_id, mode='r') as file:
        csv_read = reader(file, delimiter=',')
        ids = [row[:-1] for row in csv_read][0]
    return data_class, data_reg, ids

def header(text="", size=32, padding=0, bold=False):
    font_weight = 'normal'
    if bold:
        font_weight = 'bold'
    st.markdown(f'<p style="background-color:rgba(255, 255, 255, 0.0);color:rgba(220, 220, 220, 1.0);font-size:{size}px;border-radius:2%;padding-left: {padding}%;font-weight: {font_weight};">{text}</p>', unsafe_allow_html=True)

# load the data for the simulation
data_class_array, data_reg_array, id_list = get_data()

# Header of the file
title = st.empty()
with title.container():
    header(text="Plane engine failure prediction system",size=40)

placeholder = st.empty() # contains the columns with the plots
placeholder_2 = st.empty() 
# contains true RUL and predictions for the current sequence
results_container = st.empty()
# create a selectbox to choose the engine
# take only the first 4 chars of the selected as there are also the RUL (start/end)
selectbox = st.empty()
start_button = st.empty()
ID = selectbox.selectbox(label='Choose engine for simulation', 
                        # label_visibility='collapsed',
                        options=id_list)[:4]
ID_x = ID + '-x'
ID_y_rul_orig = ID + '-y_rul_orig'
ID_y_pred_class = ID + '-y_class_pred'
ID_y_pred_reg = ID + '-y_regr_pred'
NO_CYCLES = data_class_array[()][ID_x].shape[1]

def line_chart(df, feature_name, counter, graph):  
    '''create a chart for each sensor. df is the data of the sequence for all features.'''
    fig = go.Figure(px.line(df.iloc[-NO_CYCLES:], 
                    x=df.index + counter, 
                    y=feature_name, 
                    ))
    fig.update_layout(
                margin=dict(l=50, r=50, t=30, b=10),
                width=300, 
                height=200,
                paper_bgcolor=BG_PAPER_COLOR,
                plot_bgcolor='rgba(215,215,215,1)',
                title_text=feature_dict[feature_name],
                title_font_size=18,
                title_font_color=MAIN_COLOR,
                title_pad=dict(b=0,l=20,r=0,t=0)
                )
    fig.update_yaxes(title_font_color=MAIN_COLOR,
                    color=MAIN_COLOR,
                    gridcolor='#999999',
                    zerolinecolor = '#999999',
                    title="",
                    )
    fig.update_xaxes(title_font_color=MAIN_COLOR,
                    color=MAIN_COLOR,
                    gridcolor='#999999',
                    zerolinecolor = '#999999',
                    title="",)
    # changes color of chart line itself
    fig.update_traces(line_color=MAIN_COLOR, line_width=2)
    graph.write(fig)

# engine simulation
if start_button.button('Start engine simulation',key='start'):
    # delete the old not needed views
    start_button.empty()
    selectbox.empty()
    with title.container():
        header(text=f"Simulation of engine {ID}", size=32, padding=1, bold=False)
    if st.button('Back',key='back'):
        pass
    counter = 0
    text = 0
    # eventloop of the engine simulation
    while True:
        # show the real RUL and model predictions
        rul = data_class_array[()][ID_y_rul_orig][counter]
        class_pred = np.argmax(data_class_array[()][ID_y_pred_class][counter])
        reg_pred = np.round(data_reg_array[()][ID_y_pred_reg][counter],0) 
        if reg_pred >= 60:
            reg_pred = 60
        # get the data for the models as a dataframe from the saved data array
        # get for each iteration a new one
        df = pd.DataFrame(data_class_array[()][ID_x][counter]) # dict of arrays is pickled in an array-object. Access it with [()]
        df.columns = FEATURES
        # drop all features, we don't want to show
        df_plot = df.drop(columns=FEATURES[NO_SHOWN_FEATURES:])
        # create the current chart of each feature
        with placeholder.container():
            # prepare the columns in which the plots will be shown
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col7, col8, col9, col10, col11, col12 = st.columns(6)
            columns = [col1, col2, col3, col4, col5, col6, col7, col8, col9,
                 col10, col11, col12]    
            for feature, col in zip(FEATURES[:len(columns)], columns):
                line_chart(df_plot, feature, counter, col)
        # last 3 charts have to be in a seperate container (layout reasons)
        with placeholder_2.container():
            col13, col14, col15, col_table = st.columns((1, 1, 1, 3))
            columns_2 = [col13, col14, col15] 
            for feature, col in zip(FEATURES[12:15], columns_2):
                line_chart(df_plot, feature, counter, col)
            # show the predictions and the rul in a table
            fig = go.Figure(
                data=go.Table(
                    header=dict(values=['True RUL', 'Urgency class', 'EWS (pred. RUL)'], 
                        fill_color=MAIN_COLOR,
                        align='center',
                        height=50,
                        font=dict(size=23)), 
                    cells=dict(values=[rul, class_pred, reg_pred],
                        #fill_color='#ESECF6',
                        align='center',
                        height=40,
                        font=dict(size=23),
                        fill = dict(color=[
                            [BG_PLOT_COLOR if rul > 30 else 'rgba(255, 255, 102, 1)' 
                            if rul > 15 else MAIN_COLOR],
                            [BG_PLOT_COLOR  if class_pred == 0 else 'rgba(255, 255, 102, 1)' 
                            if class_pred == 1 else MAIN_COLOR],
                            [BG_PLOT_COLOR  if reg_pred > 59 else 'rgba(255, 255, 102, 1)'
                            if reg_pred > 15 else MAIN_COLOR]],#unique color for the first column
                        ))))
            fig.update_layout(
                title_text='Engine failure prediction',
                title_font_size=30,
                title_font_color=MAIN_COLOR,
                title_pad=dict(b=0,l=15,r=0,t=0),
                margin=dict(l=55, r=15, t=45, b=0),
                height=160,
                width=800,
                paper_bgcolor= BG_PAPER_COLOR 
                )
            col_table.write(fig)

        # if 59 is reached for the first time make maintenance announcement
        # text: 0 for 60 cycles, 1 for 30 cycles, 2 for 15 cycles
        if text == 0 and reg_pred < 60:
            text = 1
            playsound('reg60.mp3')
        elif text == 1 and class_pred == 1:
            text = 2
            playsound('class1.mp3')
        elif text == 2 and class_pred == 2:
            text = 3
            playsound('class2.mp3')  

        counter += 1
        # stops when there are no more sequences
        if counter >= data_class_array[()][ID_x].shape[0]:
            st.write("END OF TIMESERIES")
            break
        # slow down the simulation once there are less then 36 cycles of RUL
        if rul <= 35:
            time.sleep(0.5)
        placeholder.empty()
        
