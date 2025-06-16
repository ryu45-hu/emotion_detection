import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Emosi Teks",
    page_icon="😊",
    layout="centered"
)

# --- Fungsi untuk Memuat Model dan Tokenizer ---
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model_and_tokenizer():
    """Memuat model H5 dan tokenizer dari file."""
    try:
        model = tf.keras.models.load_model('model_emosi.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error saat memuat model atau tokenizer: {e}")
        return None, None

# Memuat model
model, tokenizer = load_model_and_tokenizer()

# --- Antarmuka Aplikasi ---
st.title("😊 Aplikasi Deteksi Emosi Teks")
st.write(
    "Masukkan sebuah kalimat dalam bahasa Indonesia, dan aplikasi akan "
    "mendeteksi apakah emosinya cenderung **Positif**, **Negatif**, atau **Netral**."
)

# Label emosi
nama_label = ['Negatif 😠', 'Netral 😐', 'Positif 😊']
# Parameter yang sama dengan saat training
max_length = 20
padding_type='post'
trunc_type='post'

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks di sini:", height=100, placeholder="Contoh: Saya sangat senang hari ini!")

# Tombol untuk prediksi
if st.button("Deteksi Emosi", use_container_width=True):
    if model is not None and tokenizer is not None:
        if user_input:
            # 1. Preprocessing teks input
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            
            # 2. Melakukan prediksi
            prediction = model.predict(padded_sequence)
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = nama_label[predicted_class_index]
            
            # 3. Menampilkan hasil
            st.subheader("Hasil Deteksi:")
            if predicted_class_index == 0: # Negatif
                st.error(f"Emosi terdeteksi: **{predicted_class_label}**")
            elif predicted_class_index == 1: # Netral
                st.info(f"Emosi terdeteksi: **{predicted_class_label}**")
            else: # Positif
                st.success(f"Emosi terdeteksi: **{predicted_class_label}**")

            # Menampilkan probabilitas (opsional)
            st.write("Probabilitas:")
            st.text(f"Negatif: {prediction[0][0]:.2%}")
            st.text(f"Netral:  {prediction[0][1]:.2%}")
            st.text(f"Positif: {prediction[0][2]:.2%}")

        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        st.error("Model tidak dapat dimuat. Pastikan file 'model_emosi.h5' dan 'tokenizer.pickle' ada di folder yang sama.")

st.markdown("---")
st.write("Dibuat dengan TensorFlow & Streamlit.")

