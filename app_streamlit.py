import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('modelo_frutas.h5')

CLASES = ['freshapples', 'freshbanana', 'freshoranges',
          'rottenapples', 'rottenbanana', 'rottenoranges']

st.title("🍎 Detector de Frescura de Frutas")
st.write("Sube una imagen de manzana, banana o naranja para detectar si está fresca o dañada.")

imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if imagen:
    img = Image.open(imagen).convert('RGB')
    st.image(img, caption="Imagen subida", use_container_width=True)

    img_resized = img.resize((100, 100))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    clase_idx = np.argmax(pred)
    clase = CLASES[clase_idx]
    confianza = float(pred[0][clase_idx]) * 100

    if 'fresh' in clase:
        st.success(f"🟢 Fruta Fresca — {clase} ({confianza:.1f}%)")
    else:
        st.error(f"🔴 Fruta Dañada — {clase} ({confianza:.1f}%)")

    st.subheader("Probabilidades por clase")
    for i, c in enumerate(CLASES):
        st.progress(float(pred[0][i]), text=f"{c}: {pred[0][i]*100:.1f}%")