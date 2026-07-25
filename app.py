import streamlit as st
import joblib
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===========================
# Load Model
# ===========================

model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# ===========================
# Inisialisasi
# ===========================

stop_factory = StopWordRemoverFactory()
stopwords = stop_factory.get_stop_words()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# ===========================
# Preprocessing
# ===========================

def preprocessing(text):

    text = text.lower()

    text = re.sub(r'&amp;?', ' ', text)

    normalisasi = {
        "yg":"yang",
        "gk":"tidak",
        "ga":"tidak",
        "gak":"tidak",
        "nggak":"tidak",
        "tdk":"tidak",
        "kagak":"tidak",
        "bgt":"sangat",
        "dg":"dengan",
        "jgn":"jangan",
        "sprti":"seperti",
        "gt":"begitu",
        "dlm":"dalam",
        "bnyk":"banyak",
        "mcm":"macam",
        "jg":"juga",
        "tp":"tapi",
        "utk":"untuk",
        "krn":"karena",
        "gmn":"gimana",
        "dn":"dan",
        "dr":"dari",
        "lbh":"lebih",
        "stlh":"setelah",
        "thn":"tahun",
        "dgn":"dengan",
        "sma":"sama",
        "sm":"sama",
        "gue":"saya",
        "deket":"dekat",
        "skr":"sekarang",
        "klo":"apabila",
        "kpd":"kepada",
        "aja":"saja",
        "udh":"sudah",
        "pk":"pakai",
        "jd":"jadi",
        "msh":"masih",
        "blm":"belum",
        "trs":"terus",
        "bs":"bisa",
        "bkn":"bukan",
        "dpt":"dapat",
        "tiap":"setiap",
        "bikin":"membuat",
        "wowo":"prabowo"
    }

    for k, v in normalisasi.items():
        text = re.sub(r'\b' + k + r'\b', v, text)

    text = re.sub(r'http\S+|www\.\S+', '', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\b\d+[a-zA-Z]+\b', '', text)
    text = re.sub(r'\b[a-zA-Z]+\d+\b', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenizing
    tokens = text.split()

    # Stopword
    tokens = [w for w in tokens if w not in stopwords]

    # Stemming
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

# ===========================
# Streamlit
# ===========================

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Analisis Sentimen")
st.write("Masukkan teks untuk mengetahui hasil analisis sentimen.")

text = st.text_area("Masukkan Teks", height=180)

if st.button("Prediksi"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:

        hasil = preprocessing(text)

        vector = tfidf.transform([hasil])

        prediksi = model.predict(vector)[0]

        if prediksi == "positif":
            st.success("😊 Sentimen Positif")

        elif prediksi == "negatif":
            st.error("😞 Sentimen Negatif")

        else:
            st.info("😐 Sentimen Netral")
