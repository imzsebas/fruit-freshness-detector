import gradio as gr

import numpy as np

from tensorflow.keras.models import load_model

from PIL import Image



model = load_model('modelo_frutas.h5')

CLASES = ['freshapples', 'freshbanana', 'freshoranges', 

          'rottenapples', 'rottenbanana', 'rottenoranges']



def predecir(imagen):

    if imagen is None:

        return {c: 0.0 for c in CLASES}, "⚠️ Por favor carga o toma una foto primero."

    

    img = Image.open(imagen).convert('RGB').resize((100, 100))

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

    inputs=gr.Image(

        sources=["upload"],

        type="filepath",

        label="imagen"

    ),

    outputs=[

        gr.Label(num_top_classes=6, label="Probabilidades"),

        gr.Textbox(label="Resultado")

    ],

    title="Detector de Frescura de Frutas",

    description="Toca el área de imagen, toma la foto y dale Submit.",

    head="""

    <script>

    function patchInputs() {

        const inputs = document.querySelectorAll('input[type="file"]');

        if (inputs.length === 0) {

            setTimeout(patchInputs, 500);

            return;

        }

        inputs.forEach(function(input) {

            input.setAttribute('capture', 'environment');

            input.setAttribute('accept', 'image/*');

        });

    }

    document.addEventListener('DOMContentLoaded', patchInputs);

    setTimeout(patchInputs, 2000);

    </script>

    """

)



if __name__ == "__main__":

    demo.launch()