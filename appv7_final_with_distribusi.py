import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.impute import SimpleImputer
import io
import chardet
import base64
import matplotlib.pyplot as plt

st.set_page_config(page_title="Universal Classification App", layout="wide")
st.title("🧠 Universal Classification App dengan SMOTE dan NearMiss")
st.markdown("Aplikasi ini memungkinkan Anda mengunggah dataset sendiri, memilih target, dan menjalankan model klasifikasi Random Forest dengan penanganan imbalance.")

def detect_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    return result['encoding'], raw_data

def read_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        encoding, raw_data = detect_encoding(uploaded_file)
        try:
            df = pd.read_csv(io.BytesIO(raw_data), encoding=encoding)
        except Exception as e:
            st.error(f"Gagal membaca file dengan encoding terdeteksi: {encoding}")
            return None
    return df

def remove_unique_id_columns(df):
    for col in df.columns:
        if ('id' in col.lower() or 'ID' in col) and df[col].nunique() == len(df):
            df.drop(columns=[col], inplace=True)
    return df

def impute_missing(df):
    imp = SimpleImputer(strategy='most_frequent')
    imputed = imp.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns[:imputed.shape[1]])

uploaded_file = st.sidebar.file_uploader("📂 Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = read_csv(uploaded_file)
    if df is not None:
        df = remove_unique_id_columns(df)
        df.replace(r"^\s*-\s*$", np.nan, regex=True, inplace=True)
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtypes == 'object' else col)
        df = df.convert_dtypes()
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                df[col] = df[col].astype(str)
        st.subheader("📌 Jumlah Nilai Kosong (Sebelum Imputasi)")
        st.write(df.isnull().sum())

        tab1, tab2, tab3 = st.tabs(["📊 Dataset", "⚙️ Pelatihan Model", "🔍 Prediksi Manual"])
        with tab1:
            st.subheader("📋 Pratinjau Dataset")
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                    df[col] = df[col].astype(str)

            st.dataframe(df.head())
            st.markdown("**Jumlah Baris:** {} | **Jumlah Kolom:** {}".format(df.shape[0], df.shape[1]))
            st.markdown("**Kolom yang tersedia:**")
            st.write(list(df.columns))

            st.subheader("📌 Statistik Deskriptif")
            desc_df = df.describe(include='all')
            desc_df = desc_df.astype(object).where(pd.notna(desc_df), '-')
            st.dataframe(desc_df)

        with tab2:
            st.subheader("⚙️ Pelatihan Model")
            target_col = st.selectbox("🎯 Pilih kolom target (label)", df.columns)

            if target_col:
                X = df.drop(columns=[target_col])
                y = df[target_col]

                label_encoders = {}
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le

                label_encoder_y = None
                label_mapping = None

                if y.dtype == 'object':
                    label_encoder_y = LabelEncoder()
                    y = label_encoder_y.fit_transform(y.astype(str))
                    label_mapping = dict(zip(range(len(label_encoder_y.classes_)), label_encoder_y.classes_))
                else:
                    label_mapping = {i: str(i) for i in np.unique(y)}

                # Filter hanya kolom kategori yang tidak semuanya kosong
valid_categorical_cols = [col for col in X.select_dtypes(include='object').columns if X[col].notna().any()]
if valid_categorical_cols:
    X = impute_missing(X)
    st.subheader("📌 Jumlah Nilai Kosong Setelah Imputasi")
    st.write(pd.DataFrame(X).isnull().sum())

    balancing_method = st.radio("🔄 Pilih metode penanganan imbalance:", ["Tanpa Balancing", "SMOTE", "NearMiss"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    try:
                    if balancing_method == "SMOTE":
                        if len(np.unique(y_train)) < 2 or min(np.bincount(y_train)) < 2:
                            st.warning("Data terlalu tidak seimbang untuk SMOTE. Gunakan metode lain.")
                        else:
                            smote = SMOTE()
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                            st.subheader("📊 Distribusi Kelas Setelah Penyeimbangan")
                            st.write(pd.Series(y_train).value_counts().rename_axis("Kelas").reset_index(name="Jumlah"))
                    elif balancing_method == "NearMiss":
                        if len(np.unique(y_train)) < 2 or min(np.bincount(y_train)) < 2:
                            st.warning("Data terlalu tidak seimbang untuk NearMiss. Gunakan metode lain.")
                        else:
                            nm = NearMiss()
                            X_train, y_train = nm.fit_resample(X_train, y_train)
                            st.subheader("📊 Distribusi Kelas Setelah Penyeimbangan")
                            st.write(pd.Series(y_train).value_counts().rename_axis("Kelas").reset_index(name="Jumlah"))
    except ValueError as ve:
                    st.error(f"Gagal dalam penyeimbangan data: {ve}")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

                st.subheader("📈 Evaluasi Model")
                st.write("**Confusion Matrix:**")
                st.write(confusion_matrix(y_test, y_pred))
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred, target_names=list(label_mapping.values()), zero_division=0))

                if y_prob is not None:
                    auc = roc_auc_score(y_test, y_prob)
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    st.write(f"**ROC AUC Score:** {auc:.4f}")
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc='lower right')
                    st.pyplot(fig)

                st.subheader("📌 Feature Importance")
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = X.columns

                fig2, ax2 = plt.subplots()
                ax2.barh(range(len(indices)), importances[indices], align='center')
                ax2.set_yticks(range(len(indices)))
                ax2.set_yticklabels([feature_names[i] for i in indices])
                ax2.invert_yaxis()
                ax2.set_xlabel('Importance')
                ax2.set_title('Feature Importance')
                st.pyplot(fig2)

        with tab3:
            st.subheader("🔍 Prediksi Manual")
            input_data = {}
            encoders = {}
            for col in df.drop(columns=[target_col]).columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    encoders[col] = le

            with st.form("manual_prediction_form"):
                cols = st.columns(2)
                for i, col in enumerate(X.columns):
                    with cols[i % 2]:
                        if col in encoders:
                            options = list(encoders[col].classes_)
                            input_data[col] = st.selectbox(f"{col}", options)
                        else:
                            input_data[col] = st.text_input(f"{col}", "0")

                submitted = st.form_submit_button("Lakukan Prediksi")

            if submitted:
                try:
                    input_df = pd.DataFrame([input_data])
                    for col in input_df.columns:
                        if col in encoders:
                            input_df[col] = encoders[col].transform(input_df[col].astype(str))
                        else:
                            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                    input_df = impute_missing(input_df)
                    input_df = input_df[X.columns]
                    prediction = model.predict(input_df)
                    predicted_label = label_mapping.get(prediction[0], str(prediction[0]))
                    st.success(f"✅ Hasil Prediksi: {predicted_label}")
                except Exception as e:
                    st.error(f"❌ Gagal memproses input: {e}")
