import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('startUp (1).csv')

#st.title('START UP PROFIT PREDICTOR APP')
#st.subheader('Built By Salmon Crushers')

st.markdown("<h1 style = 'color: #38419D; text-align: center; font-size: 60px; font-family: Helvetica'>START UP PROFIT PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Salmon Crushers Daintree</h4>", unsafe_allow_html = True)


st.markdown("<br>", unsafe_allow_html=True)

#add image
st.image('pngwing.com (2).png', caption ='Built by Eme Ita')

#add project problem statement
st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("<p>By analyzing a diverse set of parameters, including Market Expense, Administrative Expense, and Research and Development Spending, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success</p>", unsafe_allow_html = True)

st.sidebar.image('pngwing.com (3).png') #to take your image to the sidebar(side bar designs)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)


st.sidebar.markdown("<br>", unsafe_allow_html=True)#sidebar break *2lines
st.sidebar.markdown("<br>", unsafe_allow_html=True)

#user inputs
rd_spend = st.sidebar.number_input('Research and Development Expense', data['R&D Spend'].min(), data['R&D Spend'].max())
admin = st.sidebar.number_input('Administration Expense', data['Administration'].min(), data['Administration'].max())
mkt = st.sidebar.number_input('Marketing Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max())
state = st.sidebar.selectbox('Company Location', data['State'].unique())

#import Transformers
admin_scaler = joblib.load('Administration_scaler.pkl')
mkt_scaler = joblib.load('Marketing Spend_scaler.pkl')
rd_spend_scaler = joblib.load('R&D Spend_scaler.pkl')
state_encoder = joblib.load('state_encoder.pkl')

user_input = pd.DataFrame()
user_input['R&D Spend'] = [rd_spend]
user_input['Administration'] = [admin]
user_input['Marketing Spend'] = [mkt]
user_input['State'] = [state]

st.markdown("<br>", unsafe_allow_html = True)
st.header('Input Variable')
st.dataframe(user_input, use_container_width = True)

# transform users input according to training scale and encoding
user_input['R&D Spend'] = rd_spend_scaler.transform(user_input[['R&D Spend']])
user_input['Administration'] = admin_scaler.transform(user_input[['Administration']])
user_input['Marketing Spend'] = mkt_scaler.transform(user_input[['Marketing Spend']])
user_input['State'] = state_encoder.transform(user_input[['State']])

st.header('Transformed Input Variable')
st.dataframe(user_input, use_container_width = True)

#Modelling ---
model = joblib.load('startUpModel.pkl')

if st.button('Predict Profitability'):
    predicted_profit = model.predict(user_input)
    st.success(f"Your Company's Predicted Profit Is {predicted_profit[0].round(2)}")
    st.snow()