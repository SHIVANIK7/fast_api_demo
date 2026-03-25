import streamlit as st
import requests
import pandas as pd

SOURCES_API = "http://127.0.0.1:8000/sources"
DESTINATIONS_API = "http://127.0.0.1:8000/destinations"
PREDICT_API = "http://127.0.0.1:8000/predict"

weeks = [
    "2026-03-30 - 2026-04-05",
    "2026-04-06 - 2026-04-12",
    "2026-04-13 - 2026-04-19",
    "2026-04-20 - 2026-04-26",
    "2026-04-27 - 2026-05-03",
    "2026-05-04 - 2026-05-10",
    "2026-05-11 - 2026-05-17",
    "2026-05-18 - 2026-05-24",
    "2026-05-25 - 2026-05-31",
    "2026-06-01 - 2026-06-07",
    "2026-06-08 - 2026-06-14",
    "2026-06-15 - 2026-06-21",
    "2026-06-22 - 2026-06-28",
]

st.set_page_config(page_title="Flight Search", layout="wide")
st.title("✈️ Flight Price Prediction")


def get_sources():
    try:
        res = requests.get(SOURCES_API, timeout=10)
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []


def get_destinations(origin=None):
    try:
        params = {"origin": origin} if origin else None
        res = requests.get(DESTINATIONS_API, params=params, timeout=10)
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []


sources = get_sources()

col1, col2, col3 = st.columns(3)

with col1:
    origin = st.selectbox("Origin", options=sources, index=None)

destinations = get_destinations(origin) if origin else []

with col2:
    destination = st.selectbox(
        "Destination",
        options=destinations,
        index=None,
        disabled=origin is None
    )

with col3:
    travel_date = st.selectbox("Week of travel", options=weeks, index=None)

if st.button("Search"):
    if not origin or not destination or not travel_date:
        st.warning("Please select origin, destination, and travel week.")
    else:
        payload = {
            "origin": origin,
            "destination": destination,
            "travel_date": travel_date.split(" - ")[0],
        }

        try:
            response = requests.post(PREDICT_API, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()

                table_df = pd.DataFrame(
                    [{
                        "Origin": data["origin"],
                        "Destination": data["destination"],
                        "Travel Date": data["date"],
                        "Predicted Price (USD)": f"${data['predicted_price']}",
                        "Airline": data["airline"],
                    }]
                )

                st.success("Prediction successful")
                st.dataframe(table_df, use_container_width=True)

            else:
                st.error(f"Prediction error: {response.status_code} {response.text}")

        except Exception as e:
            st.error(f"Prediction API failed: {e}")