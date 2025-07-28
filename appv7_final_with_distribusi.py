import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Klasifikasi Dataset Imbalance dengan Random Forest", layout="wide")
st.title("Aplikasi Klasifikasi Dataset Imbalance")

# Fungsi imputasi missing value
def impute_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    return df

# Fungsi encode label
def encode_categorical(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# Fungsi training dan evaluasi

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report, y_test, y_pred

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
if uploaded_file is not None:
    # Baca file
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
    except:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    st.subheader("Dataset")
    st.write(df.head())

    df = impute_missing(df)

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom target", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Validasi dan encode
        valid_categorical_cols = X.select_dtypes(include=['object']).columns
        if len(valid_categorical_cols) > 0:
            X = encode_categorical(X)
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        tab1, tab2, tab3 = st.tabs(["Tanpa Balancing", "SMOTE", "NearMiss"])

        with tab1:
            st.markdown("### Tanpa Balancing")
            report, y_test, y_pred = train_and_evaluate(X, y)
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report")
            st.write(pd.DataFrame(report).transpose())

        with tab2:
            st.markdown("### SMOTE")
            sm = SMOTE(random_state=42)
            X_sm, y_sm = sm.fit_resample(X, y)
            report, y_test, y_pred = train_and_evaluate(X_sm, y_sm)
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report")
            st.write(pd.DataFrame(report).transpose())

        with tab3:
            st.markdown("### NearMiss")
            nm = NearMiss()
            X_nm, y_nm = nm.fit_resample(X, y)
            report, y_test, y_pred = train_and_evaluate(X_nm, y_nm)
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report")
            st.write(pd.DataFrame(report).transpose())

    st.markdown("---")
    st.subheader("Prediksi Manual")
    input_data = {}
    for col in df.columns:
        if col != target_col:
            val = st.text_input(f"{col}", value=str(df[col].iloc[0]))
            input_data[col] = float(val)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        if len(valid_categorical_cols) > 0:
            input_df = encode_categorical(input_df)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        pred = model.predict(input_df)
        st.success(f"Hasil Prediksi: {pred[0]}")
