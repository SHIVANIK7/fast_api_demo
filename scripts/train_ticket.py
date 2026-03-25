import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle
import joblib

def load_and_prepare_data(df):
    df = df.copy()
    
    df["date"] = pd.to_datetime(df["date"])
    
    # Time features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    
    return df

def encode_locations(df):
    origin_encoder = LabelEncoder()
    dest_encoder = LabelEncoder()
    
    df["origin_encoded"] = origin_encoder.fit_transform(df["origin"])
    df["destination_encoded"] = dest_encoder.fit_transform(df["destination"])
    
    return df, origin_encoder, dest_encoder

def safe_label_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1  # unseen category
    
def add_lag_features(df):
    df = df.copy()
    df["route"] = df["origin"] + "_" + df["destination"]
    
    df = df.sort_values(["route", "date"])
    
    for lag in [1, 2, 3]:
        df[f"price_lag_{lag}"] = df.groupby("route")["price"].shift(lag)
    
    df["rolling_mean_3"] = (
        df.groupby("route")["price"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    
    return df

def train_model(df):
    df_train = df[
        (df["date"] >= "2024-12-01") &
        (df["date"] <= "2026-03-01")
    ].copy()
    
    df_train = add_lag_features(df_train)
    df_train = df_train.dropna()
    
    features = [
        "origin_encoded",
        "destination_encoded",
        "month",
        "year",
        "quarter",
        "price_lag_1",
        "price_lag_2",
        "price_lag_3",
        "rolling_mean_3"
    ]
    
    target = "price"
    
    # Time split
    train_data = df_train[df_train["date"] < "2026-01-01"]
    val_data = df_train[df_train["date"] >= "2026-01-01"]
    
    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Validation RMSE: {rmse:.2f}")
    
    return model, features

def create_airline_lookup(df):
    df["route"] = df["origin"] + "_" + df["destination"]
    return df.groupby("route")["airline"].first().to_dict()

def create_future_dataframe(origin, destination, start_date, weeks=12):
    dates = pd.date_range(start=start_date, periods=weeks, freq="W")
    
    df_future = pd.DataFrame({"date": dates})
    
    df_future["origin"] = origin
    df_future["destination"] = destination
    
    df_future["month"] = df_future["date"].dt.month
    df_future["year"] = df_future["date"].dt.year
    df_future["quarter"] = df_future["date"].dt.quarter
    
    df_future["month_sin"] = np.sin(2 * np.pi * df_future["month"] / 12)
    df_future["month_cos"] = np.cos(2 * np.pi * df_future["month"] / 12)
    
    return df_future

def predict_prices(
    model,
    df,
    origin,
    destination,
    origin_encoder,
    dest_encoder,
    features,
    airline_lookup,
    start_date="2026-04-01"
):
    df = df.copy()
    
    route = f"{origin}_{destination}"
    
    # Encode safely
    origin_encoded = safe_label_transform(origin_encoder, origin)
    destination_encoded = safe_label_transform(dest_encoder, destination)
    
    # History for lag features
    history = df[
        (df["origin"] == origin) &
        (df["destination"] == destination)
    ].sort_values("date").copy()
    
    if len(history) < 3:
        raise ValueError("Not enough historical data for this route")
    
    predictions = []
    
    future_df = create_future_dataframe(origin, destination, start_date)
    
    for _, row in future_df.iterrows():
        temp = {}
        
        temp["origin_encoded"] = origin_encoded
        temp["destination_encoded"] = destination_encoded
        temp["month"] = row["month"]
        temp["year"] = row["year"]
        temp["quarter"] = row["quarter"]
        temp["month_sin"] = row["month_sin"]
        temp["month_cos"] = row["month_cos"]
        
        temp["price_lag_1"] = history["price"].iloc[-1]
        temp["price_lag_2"] = history["price"].iloc[-2]
        temp["price_lag_3"] = history["price"].iloc[-3]
        
        temp["rolling_mean_3"] = history["price"].iloc[-3:].mean()
        
        X = pd.DataFrame([temp])[features]
        
        pred_price = model.predict(X)[0]
        
        predictions.append({
            "date": row["date"],
            "predicted_price": float(pred_price),
            "airline": airline_lookup.get(route, "Unknown")
        })
        
        # Recursive update
        new_row = pd.DataFrame({
            "origin": [origin],
            "destination": [destination],
            "date": [row["date"]],
            "price": [pred_price]
        })
        
        history = pd.concat([history, new_row], ignore_index=True)
    
    return pd.DataFrame(predictions)

def save_artifacts(
    model,
    origin_encoder,
    dest_encoder,
    features,
    airline_lookup,
    df,   # <-- ADD THIS
    artifacts_dir="artifacts"
):
    import os
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(model, f"{artifacts_dir}/model.pkl")
    joblib.dump(origin_encoder, f"{artifacts_dir}/origin_encoder.pkl")
    joblib.dump(dest_encoder, f"{artifacts_dir}/dest_encoder.pkl")
    joblib.dump(features, f"{artifacts_dir}/features.pkl")
    joblib.dump(airline_lookup, f"{artifacts_dir}/airline_lookup.pkl")

    # ✅ SAVE HISTORY
    df.to_parquet(f"{artifacts_dir}/data.parquet", index=False)
    
def main():
    df = pd.read_csv(r"C:\FastAPIDemo\fast_api_demo\scripts\data.csv")
    
    df = load_and_prepare_data(df)
    df, origin_encoder, dest_encoder = encode_locations(df)
    
    model, features = train_model(df)
    airline_lookup = create_airline_lookup(df)
    
    save_artifacts(model, origin_encoder, dest_encoder, features, airline_lookup, df)  # <-- PASS df
    
    # Example prediction
    pred_df = predict_prices(
        model,
        df,
        origin="Dubai",
        destination="London",
        origin_encoder=origin_encoder,
        dest_encoder=dest_encoder,
        features=features,
        airline_lookup=airline_lookup
    )
    
    print(pred_df.head())


if __name__ == "__main__":
    main()