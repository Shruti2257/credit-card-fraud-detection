import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# -------------------------------
# LOAD MODEL + SCALER + DATASET
# -------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_resource
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

model, scaler = load_assets()
df = load_data()

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Dashboard", "Predict Transaction", "Feature Importance", "Batch CSV Prediction", "View Dataset"]
)

# ---------------------------------------------------------
# PAGE 1 — DASHBOARD
# ---------------------------------------------------------
if page == "Dashboard":

    st.title("📊 Credit Card Fraud Detection Dashboard")

    # KPIs
    total = len(df)
    frauds = int(df["Class"].sum())
    fraud_rate = round((frauds / total) * 100, 4)

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Transactions", f"{total:,}")
    k2.metric("Fraud Cases", f"{frauds:,}")
    k3.metric("Fraud Rate", f"{fraud_rate}%")

    st.markdown("---")

    # CLASS HISTOGRAM
    st.subheader("Class Distribution (Histogram)")
    fig1 = px.histogram(df, x="Class", color="Class", title="0 = Normal, 1 = Fraud")
    st.plotly_chart(fig1, use_container_width=True)

    # AMOUNT HISTOGRAM
    st.subheader("Amount Distribution (Log Scale Histogram)")
    fig2 = px.histogram(df, x="Amount", nbins=60, title="Transaction Amount Distribution")
    fig2.update_layout(yaxis_type="log")
    st.plotly_chart(fig2, use_container_width=True)

    # SCATTER
    st.subheader("Time vs Amount Scatter")
    sample = df.sample(min(len(df), 4000), random_state=42)
    fig3 = px.scatter(sample, x="Time", y="Amount", color="Class", title="Time vs Amount (Sample)")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2 — PREDICT TRANSACTION (Manual + Auto-load)
# ---------------------------------------------------------
# ---------------------------------------------------------
# PAGE 2 – PREDICT TRANSACTION (COMPLETE FIX)
# ---------------------------------------------------------
elif page == "Predict Transaction":

    st.title("🔍 Predict Fraud for a Single Transaction")
    
    # Two tabs: Load Sample OR Manual Entry
    tab1, tab2 = st.tabs(["📂 Load Real Sample", "✍️ Manual Entry (All Features)"])
    
    # ============ TAB 1: LOAD SAMPLE ============
    with tab1:
        st.write("Load a real transaction from the dataset and predict instantly.")
        
        col1, col2 = st.columns(2)
        
        if col1.button("📘 Load Normal Sample", use_container_width=True):
            normal_sample = df[df["Class"] == 0].sample(1, random_state=42)
            full_sample = normal_sample[FEATURE_ORDER]
            
            X_scaled = scaler.transform(full_sample)
            proba = model.predict_proba(X_scaled)[0][1]
            pred = int(proba >= 0.50)

            st.success("✅ Loaded REAL Normal Transaction")
            st.metric("Fraud Probability", f"{proba*100:.2f}%")

            if pred == 1:
                st.error("🚨 Model predicted FRAUD!")
            else:
                st.success("✅ Model predicted NORMAL")

            with st.expander("📊 View Transaction Data"):
                st.dataframe(full_sample)
        
        if col2.button("🚨 Load Fraud Sample", use_container_width=True):
            fraud_sample = df[df["Class"] == 1].sample(1, random_state=42)
            full_sample = fraud_sample[FEATURE_ORDER]
            
            X_scaled = scaler.transform(full_sample)
            proba = model.predict_proba(X_scaled)[0][1]
            pred = int(proba >= 0.50)

            st.warning("⚠️ Loaded REAL Fraud Transaction")
            st.metric("Fraud Probability", f"{proba*100:.2f}%")

            if pred == 1:
                st.error("🚨 Model predicted FRAUD!")
            else:
                st.success("✅ Model predicted NORMAL")

            with st.expander("📊 View Transaction Data"):
                st.dataframe(full_sample)
    
    # ============ TAB 2: MANUAL ENTRY (ALL 30 FEATURES) ============
    with tab2:
        st.info("💡 **Tip:** To test with real fraud values, go to 'View Dataset' page, filter Class=1, and copy values here.")
        
        # Initialize session state for form values
        if 'form_values' not in st.session_state:
            st.session_state.form_values = df[FEATURE_ORDER].median().to_dict()
        
        # Quick load buttons
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("📥 Load Fraud Values to Form"):
            fraud_sample = df[df["Class"] == 1].sample(1, random_state=42)
            st.session_state.form_values = fraud_sample[FEATURE_ORDER].iloc[0].to_dict()
            st.success("Fraud values loaded! Scroll down and click 'Predict'")
        
        if col_b.button("📥 Load Normal Values to Form"):
            normal_sample = df[df["Class"] == 0].sample(1, random_state=42)
            st.session_state.form_values = normal_sample[FEATURE_ORDER].iloc[0].to_dict()
            st.success("Normal values loaded! Scroll down and click 'Predict'")
        
        if col_c.button("🔄 Reset to Defaults"):
            st.session_state.form_values = df[FEATURE_ORDER].median().to_dict()
            st.info("Reset to median values")
        
        st.markdown("---")
        
        # Create form with ALL 30 features
        with st.form("prediction_form"):
            st.subheader("Enter All 30 Features")
            
            # Time and Amount
            col1, col2 = st.columns(2)
            time_val = col1.number_input("Time", value=float(st.session_state.form_values["Time"]), format="%.2f")
            amount_val = col2.number_input("Amount", value=float(st.session_state.form_values["Amount"]), format="%.2f")
            
            # V1 to V28 in expandable sections
            with st.expander("V1 - V10 Features", expanded=True):
                cols = st.columns(5)
                v1 = cols[0].number_input("V1", value=float(st.session_state.form_values["V1"]), format="%.6f")
                v2 = cols[1].number_input("V2", value=float(st.session_state.form_values["V2"]), format="%.6f")
                v3 = cols[2].number_input("V3", value=float(st.session_state.form_values["V3"]), format="%.6f")
                v4 = cols[3].number_input("V4", value=float(st.session_state.form_values["V4"]), format="%.6f")
                v5 = cols[4].number_input("V5", value=float(st.session_state.form_values["V5"]), format="%.6f")
                
                cols = st.columns(5)
                v6 = cols[0].number_input("V6", value=float(st.session_state.form_values["V6"]), format="%.6f")
                v7 = cols[1].number_input("V7", value=float(st.session_state.form_values["V7"]), format="%.6f")
                v8 = cols[2].number_input("V8", value=float(st.session_state.form_values["V8"]), format="%.6f")
                v9 = cols[3].number_input("V9", value=float(st.session_state.form_values["V9"]), format="%.6f")
                v10 = cols[4].number_input("V10", value=float(st.session_state.form_values["V10"]), format="%.6f")
            
            with st.expander("V11 - V20 Features"):
                cols = st.columns(5)
                v11 = cols[0].number_input("V11", value=float(st.session_state.form_values["V11"]), format="%.6f")
                v12 = cols[1].number_input("V12", value=float(st.session_state.form_values["V12"]), format="%.6f")
                v13 = cols[2].number_input("V13", value=float(st.session_state.form_values["V13"]), format="%.6f")
                v14 = cols[3].number_input("V14", value=float(st.session_state.form_values["V14"]), format="%.6f")
                v15 = cols[4].number_input("V15", value=float(st.session_state.form_values["V15"]), format="%.6f")
                
                cols = st.columns(5)
                v16 = cols[0].number_input("V16", value=float(st.session_state.form_values["V16"]), format="%.6f")
                v17 = cols[1].number_input("V17", value=float(st.session_state.form_values["V17"]), format="%.6f")
                v18 = cols[2].number_input("V18", value=float(st.session_state.form_values["V18"]), format="%.6f")
                v19 = cols[3].number_input("V19", value=float(st.session_state.form_values["V19"]), format="%.6f")
                v20 = cols[4].number_input("V20", value=float(st.session_state.form_values["V20"]), format="%.6f")
            
            with st.expander("V21 - V28 Features"):
                cols = st.columns(5)
                v21 = cols[0].number_input("V21", value=float(st.session_state.form_values["V21"]), format="%.6f")
                v22 = cols[1].number_input("V22", value=float(st.session_state.form_values["V22"]), format="%.6f")
                v23 = cols[2].number_input("V23", value=float(st.session_state.form_values["V23"]), format="%.6f")
                v24 = cols[3].number_input("V24", value=float(st.session_state.form_values["V24"]), format="%.6f")
                v25 = cols[4].number_input("V25", value=float(st.session_state.form_values["V25"]), format="%.6f")
                
                cols = st.columns(5)
                v26 = cols[0].number_input("V26", value=float(st.session_state.form_values["V26"]), format="%.6f")
                v27 = cols[1].number_input("V27", value=float(st.session_state.form_values["V27"]), format="%.6f")
                v28 = cols[2].number_input("V28", value=float(st.session_state.form_values["V28"]), format="%.6f")
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Fraud", type="primary", use_container_width=True)
        
        # Process prediction when form is submitted
        if submitted:
            # Collect all input values
            input_data = {
                "Time": time_val,
                "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
                "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
                "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
                "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
                "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
                "V26": v26, "V27": v27, "V28": v28,
                "Amount": amount_val
            }
            
            # Create DataFrame with proper feature order
            X = pd.DataFrame([input_data])[FEATURE_ORDER]
            X_scaled = scaler.transform(X)

            proba = model.predict_proba(X_scaled)[0][1]
            pred = int(proba >= 0.50)

            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Show probability with color coding
            col1, col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{proba*100:.2f}%")
            col2.metric("Prediction", "FRAUD 🚨" if pred == 1 else "NORMAL ✅")

            if pred == 1:
                st.error("🚨 **HIGH RISK:** This transaction is LIKELY FRAUDULENT!")
            else:
                st.success("✅ **LOW RISK:** This transaction appears NORMAL.")
            
            # Show input data used
            with st.expander("📊 View All Input Values Used"):
                st.dataframe(X)
# ---------------------------------------------------------
# PAGE 3 — FEATURE IMPORTANCE
# ---------------------------------------------------------
elif page == "Feature Importance":

    st.title("📊 Feature Importance (Random Forest)")

    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "importance": importance
    }).sort_values("importance", ascending=False)

    fig = px.bar(fi_df.head(20), x="importance", y="feature",
                 orientation="h", title="Top 20 Feature Importances")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fi_df)

# ---------------------------------------------------------
# PAGE 4 — BATCH CSV PREDICTION
# ---------------------------------------------------------
elif page == "Batch CSV Prediction":

    st.title("📁 Batch Fraud Prediction (Upload CSV)")

    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file:
        batch = pd.read_csv(file)

        missing = [c for c in FEATURE_ORDER if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = batch[FEATURE_ORDER]
            X_scaled = scaler.transform(X)

            batch["fraud_probability"] = model.predict_proba(X_scaled)[:, 1]
            batch["prediction"] = (batch["fraud_probability"] >= 0.50).astype(int)

            st.success("Batch prediction completed.")
            st.dataframe(batch.head(50))

            st.download_button(
                "Download Result CSV",
                batch.to_csv(index=False),
                file_name="fraud_predictions.csv"
            )

# ---------------------------------------------------------
# PAGE 5 — VIEW DATASET
# ---------------------------------------------------------
elif page == "View Dataset":

    st.title("📁 Dataset Viewer")

    rows = st.slider("Rows to display", 5, 200, 50)
    st.dataframe(df.head(rows))

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        file_name="creditcard.csv"
    )
