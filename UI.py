import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from datetime import timedelta

st.title("📈 Demand Accuracy & Forecast with Prophet")

# --- Section 1: Demand Accuracy Prediction ---
st.header("🛒 Demand Accuracy Checker")
with st.form("accuracy_form"):
    store_id = st.text_input("Store ID", "S001")
    product_id = st.text_input("Product ID", "P0001")
    category = st.text_input("Category (numeric code)", "0")
    region = st.text_input("Region (numeric code)", "0")
    inventory_level = st.number_input("Inventory Level", value=200)
    units_sold = st.number_input("Units Sold", value=130)
    units_ordered = st.number_input("Units Ordered", value=150)
    demand_forecast = st.number_input("Demand Forecast", value=140.0)
    price = st.number_input("Price", value=35.5)
    discount = st.number_input("Discount (%)", value=10)
    weather = st.text_input("Weather Condition (numeric code)", "0")
    promotion = st.text_input("Holiday/Promotion (0/1)", "0")
    competitor_price = st.number_input("Competitor Pricing", value=30.0)
    seasonality = st.text_input("Seasonality (numeric code)", "0")

    submitted = st.form_submit_button("Check Accuracy")
    if submitted:
        rf_model = joblib.load("demand_accuracy_model.pkl")
        sample = pd.DataFrame([{
            'Store ID': 0, 'Product ID': 0, 'Category': int(category), 'Region': int(region),
            'Inventory Level': inventory_level, 'Units Sold': units_sold,
            'Units Ordered': units_ordered, 'Demand Forecast': demand_forecast,
            'Price': price, 'Discount': discount, 'Weather Condition': int(weather),
            'Holiday/Promotion': int(promotion), 'Competitor Pricing': competitor_price,
            'Seasonality': int(seasonality),
            'forecast_diff': abs(units_sold - demand_forecast),
            'sold_inventory_ratio': units_sold / inventory_level,
            'order_vs_sales_ratio': units_ordered / (units_sold + 1e-5)
        }])
        result = rf_model.predict(sample)
        st.success("✅ Demand is Accurate" if int(result[0]) == 1 else "❌ Demand seems Inaccurate")

# --- Section 2: Demand Forecast for Next 7 Days using Prophet ---
st.header("📊 Predict Next 7 Days Demand")
file = st.file_uploader("Upload 30 Days CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    product_data = df[(df['Store ID'] == 'S001') & (df['Product ID'] == 'P0001')]

    if len(product_data) >= 30:
        # Prophet Forecast
        prophet_df = product_data[['Date', 'Units Sold']].rename(columns={'Date': 'ds', 'Units Sold': 'y'})
        prophet_model = Prophet()
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=7)
        forecast = prophet_model.predict(future)
        output = forecast[['ds', 'yhat']].tail(7)
        output['yhat'] = output['yhat'].round().astype(int)
        st.subheader("📅Demand Forecast:")
        st.dataframe(output.rename(columns={'ds': 'Date', 'yhat': 'Predicted Units'}))
    else:
        st.warning("Please upload at least 30 days of data for a single Store ID and Product ID combo.")
