import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Load dataset
st.set_page_config(layout="wide")
st.title("Aplikasi Klasifikasi dengan Random Forest")
data = pd.read_csv("diabetes.csv")

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Halaman", ["Dataset", "Preprocessing", "Model", "Distribusi", "Evaluasi"])

# Tab1 - Dataset
if menu == "Dataset":
    st.header("Dataset")
    st.dataframe(data)

# Tab2 - Preprocessing
elif menu == "Preprocessing":
    st.header("Preprocessing")
    missing_values = data.isnull().sum()
    st.subheader("Cek Missing Values")
    st.write(missing_values)
    
    st.subheader("Encoding jika perlu (LabelEncoder untuk kolom object)")
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    st.write("Data setelah encoding:")
    st.dataframe(data)

# Tab3 - Model
elif menu == "Model":
    st.header("Model Training")
    target_column = st.selectbox("Pilih Kolom Target", data.columns)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    sampling = st.radio("Pilih Metode Sampling", ["Tanpa Balancing", "SMOTE", "NearMiss"])

    if sampling == "SMOTE":
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        st.write("SMOTE Applied")
    elif sampling == "NearMiss":
        nearmiss = NearMiss()
        X_resampled, y_resampled = nearmiss.fit_resample(X, y)
        st.write("NearMiss Applied")
    else:
        X_resampled, y_resampled = X, y
        st.write("Tanpa Balancing")

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# Tab4 - Distribusi
elif menu == "Distribusi":
    st.header("Distribusi Kelas Sebelum dan Sesudah Balancing")
    tab1, tab2, tab3 = st.tabs(["Original", "SMOTE", "NearMiss"])

    with tab1:
        st.subheader("Original Class Distribution")
        st.bar_chart(data[target_column].value_counts())

    with tab2:
        sm = SMOTE()
        X_sm, y_sm = sm.fit_resample(X, y)
        st.subheader("SMOTE Class Distribution")
        st.bar_chart(pd.Series(y_sm).value_counts())

    with tab3:
        nm = NearMiss()
        X_nm, y_nm = nm.fit_resample(X, y)
        st.subheader("NearMiss Class Distribution")
        st.bar_chart(pd.Series(y_nm).value_counts())

# Tab5 - Evaluasi
elif menu == "Evaluasi":
    st.header("Evaluasi Model")
    st.markdown("Silakan jalankan pelatihan model pada tab 'Model' terlebih dahulu.")
