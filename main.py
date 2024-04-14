# main.py
import streamlit as st
from Q1_Coin_Search import main as Q1
from Q2_Coin_Comparision import main as Q2
from Q3_Number_Detect import main as Q3

def main():
    st.title('Multi-App Streamlit')

    app_selector = st.sidebar.radio("Select App", ('Coin Search', 'Compare Coin', 'Number Classify'))

    if app_selector == 'Coin Search':
        Q1()
    elif app_selector == 'Compare Coin':
        Q2()
    elif app_selector == 'Number Classify':
        Q3()

if __name__ == "__main__":
    main()
