import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to get coin data
def get_coin_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?x_cg_demo_api_key=CG-7RA7eV8yvYgJRozgko2EGnHk"
    params = {
        "vs_currency": "usd",
        "days": days
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        st.error("Error fetching data")
        st.error(response.json())

# Main function
def main():
    st.title("Cryptocurrency Comparison")

    # Fetching available coins
    coins_list_url = "https://api.coingecko.com/api/v3/coins/list?x_cg_demo_api_key=CG-7RA7eV8yvYgJRozgko2EGnHk"
    coins_list_response = requests.get(coins_list_url)
    coins_list = coins_list_response.json()
    coins_dict = {coin['id']: coin['name'] for coin in coins_list}  # Adjusted to use symbol instead of name

    # User input for coin selection
    coin_input_1 = st.selectbox("Select the first cryptocurrency", options=list(coins_dict.values()))
    coin_id_1 = [coin_id for coin_id, name in coins_dict.items() if name == coin_input_1][0]

    coin_input_2 = st.selectbox("Select the second cryptocurrency", options=list(coins_dict.values()))
    coin_id_2 = [coin_id for coin_id, name in coins_dict.items() if name == coin_input_2][0]

    # User input for time frame selection
    time_frames = {
        "1 Week": 7,
        "1 Month": 30,
        "1 Year": 365,
        "5 Years": 365 * 5
    }
    selected_time_frame = st.selectbox("Select the time frame", options=list(time_frames.keys()))

    # Fetch coin data
    coin_data_1 = get_coin_data(coin_id_1, time_frames[selected_time_frame])
    coin_data_2 = get_coin_data(coin_id_2, time_frames[selected_time_frame])

    if coin_data_1 is not None and coin_data_2 is not None:
        # Plot price comparison
        st.subheader(f"Price comparison between {coin_input_1} and {coin_input_2} over {selected_time_frame}")
        plt.figure(figsize=(10, 6))
        plt.plot(coin_data_1['timestamp'], coin_data_1['price'], label=coin_input_1)
        plt.plot(coin_data_2['timestamp'], coin_data_2['price'], label=coin_input_2)
        plt.title(f"Price Comparison Over {selected_time_frame}")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        st.pyplot()

if __name__ == "__main__":
    main()
