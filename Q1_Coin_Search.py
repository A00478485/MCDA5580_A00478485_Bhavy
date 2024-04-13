import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to get coin data
def get_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?x_cg_demo_api_key=CG-7RA7eV8yvYgJRozgko2EGnHk"
    params = {
        "vs_currency": "usd",
        "days": 365
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

# Main function
def main():
    st.title("Cryptocurrency Details")

    # Fetching available coins
    coins_list_url = "https://api.coingecko.com/api/v3/coins/list?x_cg_demo_api_key=CG-7RA7eV8yvYgJRozgko2EGnHk"
    coins_list_response = requests.get(coins_list_url)
    coins_list = coins_list_response.json()
    coins_dict = {coin['id']: coin['name'] for coin in coins_list}

    # User input for coin selection
    coin_input = st.selectbox("Select a cryptocurrency", options=list(coins_dict.values()))

    coin_id = [coin_id for coin_id, name in coins_dict.items() if name == coin_input][0]

    # Fetch coin data
    coin_data = get_coin_data(coin_id)

    if coin_data is not None:
        # Plot price over the last year
        st.subheader(f"Price chart for {coin_input}")
        plt.figure(figsize=(10, 6))
        plt.plot(coin_data['timestamp'], coin_data['price'])
        plt.title(f"{coin_input} Price Over Last Year")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        st.pyplot()

        # Print max and min prices
        max_price = coin_data['price'].max()
        min_price = coin_data['price'].min()
        st.write(f"Maximum Price: ${max_price:.2f}")
        st.write(f"Minimum Price: ${min_price:.2f}")

        # Print day with highest and lowest prices
        max_price_date = coin_data.loc[coin_data['price'].idxmax(), 'timestamp']
        min_price_date = coin_data.loc[coin_data['price'].idxmin(), 'timestamp']
        st.write(f"Day with Highest Price: {max_price_date}")
        st.write(f"Day with Lowest Price: {min_price_date}")

if __name__ == "__main__":
    main()
