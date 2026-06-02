import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('modelo_frutas.h5')

CLASES = ['freshapples', 'freshbanana', 'freshoranges', 
          'rottenapples', 'rottenbanana', 'rottenoranges']

def predecir(imagen):
    img = Image.fromarray(imagen).convert('RGB').resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    clase_idx = np.argmax(pred)
    clase = CLASES[clase_idx]
    confianza = float(pred[0][clase_idx])
    
    estado = "🟢 Fruta Fresca" if 'fresh' in clase else "🔴 Fruta Dañada"
    
    return {c: float(pred[0][i]) for i, c in enumerate(CLASES)}, f"{estado} — {clase} ({confianza*100:.1f}%)"

demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(),
    outputs=[
        gr.Label(num_top_classes=6, label="Probabilidades"),
        gr.Textbox(label="Resultado")
    ],
    title="Detector de Frescura de Frutas",
    description="Sube una imagen de manzana, banana o naranja para detectar si está fresca o dañada."
)

if __name__ == "__main__":
    demo.launch()