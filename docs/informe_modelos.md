# Informe Comparativo de Modelos
## Proyecto: Detección de Frescura de Frutas
### Aprendizaje Computacional — Ingeniería de Sistemas

---

## 1. Introducción

Este informe presenta el análisis comparativo entre dos arquitecturas de redes neuronales
convolucionales entrenadas para clasificar imágenes de frutas en seis categorías:
freshapples, freshbanana, freshoranges, rottenapples, rottenbanana y rottenoranges.

El objetivo es determinar cuál modelo ofrece mejor rendimiento para ser desplegado
en producción, considerando no solo la precisión sino también el tamaño del modelo
y el tiempo de entrenamiento.

---

## 2. Dataset

- **Nombre:** Fruits Fresh and Rotten for Classification
- **Fuente:** Kaggle (sriramr)
- **Total de imágenes:** 13,599
- **División:** Training / Test predefinida por el dataset
- **Subdivisión Training:** 80% entrenamiento, 20% validación (split interno)
- **Tamaño de imagen:** Redimensionadas a 100x100 píxeles
- **Clases:** 6 (3 frutas x 2 estados)

### Distribución por clase (Test set)

| Clase | Imágenes |
|---|---|
| freshapples | 395 |
| freshbanana | 381 |
| freshoranges | 388 |
| rottenapples | 601 |
| rottenbanana | 530 |
| rottenoranges | 403 |
| **Total** | **2,698** |

---

## 3. Arquitecturas

### 3.1 CNN Baseline

Arquitectura diseñada desde cero con 3 bloques convolucionales:

- Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
- Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
- Conv2D(128, 3x3) + ReLU + MaxPooling(2x2)
- Flatten
- Dense(128) + ReLU + Dropout(0.5)
- Dense(6) + Softmax

Optimizador: Adam | Loss: Categorical Crossentropy | Epocas: 10 | Batch: 32

### 3.2 MobileNetV2 con Transfer Learning

Arquitectura preentrenada en ImageNet con capas propias añadidas:

- Base: MobileNetV2 (pesos ImageNet, capas base congeladas)
- GlobalAveragePooling2D
- Dense(128) + ReLU + Dropout(0.3)
- Dense(6) + Softmax

Optimizador: Adam | Loss: Categorical Crossentropy | Epocas: 10 | Batch: 32

La diferencia clave es que MobileNetV2 ya conoce características visuales generales
aprendidas de millones de imágenes, mientras que la CNN baseline aprende todo
desde cero con solo 13,599 imágenes.

---

## 4. Resultados

### 4.1 Métricas generales

| Métrica | CNN Baseline | MobileNetV2 |
|---|---|---|
| Train Accuracy (epoca 10) | 95.82% | 94%+ |
| Val Accuracy (epoca 10) | 98.44% | 95%+ |
| Test Accuracy | 97% | 99% |
| F1-score macro | 0.97 | 0.99 |
| Precision macro | 0.97 | 0.99 |
| Recall macro | 0.97 | 0.98 |
| Tamaño del modelo | ~15MB | ~11.5MB |
| Tiempo de entrenamiento | ~35 min | ~30 min |

### 4.2 Reporte por clase — CNN Baseline

| Clase | Precision | Recall | F1-score | Soporte |
|---|---|---|---|---|
| freshapples | 0.94 | 0.99 | 0.97 | 395 |
| freshbanana | 0.99 | 1.00 | 1.00 | 381 |
| freshoranges | 0.98 | 0.94 | 0.96 | 388 |
| rottenapples | 0.95 | 0.96 | 0.95 | 601 |
| rottenbanana | 0.99 | 1.00 | 1.00 | 530 |
| rottenoranges | 0.96 | 0.92 | 0.94 | 403 |
| **macro avg** | **0.97** | **0.97** | **0.97** | **2698** |

### 4.3 Reporte por clase — MobileNetV2

| Clase | Precision | Recall | F1-score | Soporte |
|---|---|---|---|---|
| freshapples | 1.00 | 0.96 | 0.98 | 395 |
| freshbanana | 0.99 | 1.00 | 1.00 | 381 |
| freshoranges | 0.99 | 0.98 | 0.99 | 388 |
| rottenapples | 0.96 | 1.00 | 0.98 | 601 |
| rottenbanana | 1.00 | 1.00 | 1.00 | 530 |
| rottenoranges | 0.98 | 0.97 | 0.98 | 403 |
| **macro avg** | **0.99** | **0.98** | **0.99** | **2698** |

---

## 5. Análisis por clase

### Clase más difícil: rottenoranges

En ambos modelos, rottenoranges presentó el recall más bajo. Esto se explica porque
las naranjas en proceso de descomposición pueden mantener un color exterior naranja
similar al de las frescas, dificultando la clasificación solo por características visuales
superficiales.

### Clase más fácil: freshbanana y rottenbanana

Ambos modelos alcanzaron F1-score de 1.00 en estas clases. Las bananas presentan
cambios visuales muy marcados entre frescas (amarillo uniforme) y podridas
(manchas marrones), lo que facilita la clasificación.

### Mejora de MobileNetV2 sobre baseline

La mayor mejora se observó en freshapples, donde el baseline tenía precision de 0.94
y MobileNetV2 alcanzó 1.00. Esto indica que el transfer learning permite capturar
características más sutiles de textura y color en manzanas frescas.

---

## 6. Análisis de curvas de aprendizaje

### CNN Baseline
El modelo mostró una convergencia estable a lo largo de las 10 épocas. La validación
superó al entrenamiento desde la época 1, lo que indica que el Dropout de 0.5 fue
efectivo para regularizar el modelo y evitar overfitting.

### MobileNetV2
Al partir de pesos preentrenados, el modelo alcanzó alta precisión desde las primeras
épocas. Las curvas de loss mostraron una caída más rápida y estable que el baseline,
confirmando la ventaja del transfer learning en datasets de tamaño mediano.

---

## 7. Consideraciones de despliegue

| Criterio | CNN Baseline | MobileNetV2 |
|---|---|---|
| Tamaño en disco | ~15MB | ~11.5MB |
| Velocidad de inferencia | Alta | Alta |
| Precision en produccion | 97% | 99% |
| Compatibilidad movil | Buena | Excelente |
| Recomendado para HF Spaces | Si | Si |

MobileNetV2 es más liviano y más preciso, lo que lo convierte en la opción ideal
para despliegue en producción, especialmente en entornos con recursos limitados
como Hugging Face Spaces con CPU Basic.

---

## 8. Conclusiones

1. MobileNetV2 superó al baseline en todas las métricas evaluadas, logrando
   99% de accuracy en test frente al 97% del baseline.

2. El transfer learning demostró ser más eficiente: MobileNetV2 alcanzó mayor
   precisión en menos tiempo de entrenamiento y con menor tamaño de modelo.

3. Ambos modelos generalizaron bien sin overfitting, validado por las curvas
   de aprendizaje donde la validación se mantiene cercana al entrenamiento.

4. La clase más difícil para ambos modelos fue rottenoranges, lo que sugiere
   que futuros trabajos podrían beneficiarse de técnicas de data augmentation
   específicas para esta clase.

5. MobileNetV2 fue seleccionado como modelo final para producción por su
   combinación de alto rendimiento, tamaño reducido y velocidad de inferencia.

---

## 9. Matriz de Confusión

![Matriz de Confusión](imagenes/matriz_confusion.png)

---

## 10. Curvas ROC-AUC

| Clase | AUC |
|---|---|
| freshapples | 1.000 |
| freshbanana | 1.000 |
| freshoranges | 1.000 |
| rottenapples | 0.999 |
| rottenbanana | 1.000 |
| rottenoranges | 0.999 |

![Curvas ROC](imagenes/roc_auc.png)