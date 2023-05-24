import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('Industrial Risk',('0','0.5','1'))
    management_risk = st.sidebar.selectbox('Management Risk',('0','0.5','1'))
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility',('0','0.5','1'))
    credibility = st.sidebar.selectbox('Credibility',('0','0.5','1'))
    competitiveness = st.sidebar.selectbox('Competitiveness',('0','0.5','1'))
    operating_risk = st.sidebar.selectbox('Operating Risk',('0','0.5','1'))
    
    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk}
    
    features = pd.DataFrame(data, index = [0])
    return features 
  
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

comapany = pd.read_csv(r"C:\Users\jaevi\Desktop\Project Bankruptcy DS\bankruptcy-prevention1.csv")

X =comapany.iloc[:,:6]
Y = comapany.iloc[:,-1]
clf = LogisticRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Bankrupt' if prediction_proba[0][1] > 0.5 else 'Non-Bankrupt')

st.subheader('Prediction Probability')
st.write(prediction_proba)