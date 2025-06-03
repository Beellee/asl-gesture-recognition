# 🤟 ASL Gesture Recognition (American Sign Language)

Este proyecto ofrece un pipeline completo para el reconocimiento de gestos de Lengua de Señas Americana (ASL) basado en keypoints extraídos con MediaPipe. Incluye desde la captura de vídeos y extracción de keypoints, hasta la augmentación de datos, el preprocesamiento, el entrenamiento de modelos (LSTM y TCN) con características auxiliares (duración y fracción de manos activas), la evaluación de desempeño y el despliegue de una API para inferencia en tiempo real.

---

## Estructura General

* **app/**
  * `app.py`: expone un endpoint REST que recibe un video, extrae keypoints de manos con MediaPipe, aplica el LSTM entrenado para clasificar el gesto y devuelve el signo predicho con su confianza
  * `index.html` / `index.css` /`index.js`: frontend de la app
  * `requirements.txt`: lista de dependencias necesarias (PyTorch, FastAPI, MediaPipe, OpenCV, scikit-learn, NumPy, etc.).

* **code/**
  * `record_videos.html`: app utilizada para grabar datos de entrenamiento 
  * `train_model.ipynb`: flujo completo utilizado para entrenar LSTM y TCN.contiene: 
      - extracción de keypoints con MediaPipe
      - aumento de datos / ruido
      - metricas para cuantificar el ruido
      - detección de la "ventana activa"
      - calculo de métricas auxiliares (duracion original / proporcion de mano derecha/izquierda utilizada en el signo)
      - flujos de procesamiento de los datos para alimentarlos a cada red neuronal
      - entrenamiento y métricas
  * `documentation`: scripts para mostrar las partes mas visuales del proyecto en la memoria.

* **data/** 
  * `keypoints/`: vídeos crudos organizados por signo.
  * `keypoints_augmented/`: keypoints obtenidos tras extracción y augmentación, organizados por signo.
  * `keypoints_active_window/`: secuencias recortadas según la ventana de movimiento activo para cada clip (sin forzar longitud).
  * `keypoints_fixed/`: secuencias normalizadas y ajustadas a longitud fija (por defecto 73 frames), listas para entrenar la LSTM.
  * `keypoints_metadata/`: archivos JSON que describen metadatos de cada secuencia (duración original, índices de inicio/fin de la ventana activa, longitud objetivo).

  (por motivos de privacidad esta carpeta no se comparte)

* **shared\_elements/**
  * `lstm.py`: definición de la clase `SignLSTM`, que consume secuencias de keypoints, duración y fracciones de manos para generar logits de clase.
  * `tcn.py`: definición de la clase `SignTCN`, con bloques de convoluciones causales dilatadas, pooling global y fusión de características auxiliares.
  * `best_model.pt`: mejor modelo de LSTM encontrado durante el entrenamiento.
  * `best_model_tcn.pt`: mejor modelo de TCN encontrado durante el entrenamiento.
  * `shared_utils.py`: utilidades compartidas (instanciación de MediaPipe, extracción de keypoints, detección de ventana activa, ajuste de longitud, cálculo de fracciones de manos activas).
  * `X_mean.npy` / `X_std.npy`: vectores de media y desviación estándar usados para normalizar las características de cada frame.
  * `label_encoder.pkl`: objeto de codificación de etiquetas (mapa entre índice y nombre de signo).



---

## Descripción del Pipeline

1. **Extracción de Keypoints**

   * Se capturan vídeos de cada signo con cámara frontal.
   * Se procesan mediante MediaPipe Holistic para obtener únicamente los keypoints de manos (cada frame resulta en un array de 42 puntos con coordenadas (x, y, z)).
   * Se guardan como archivos `.npy` organizados en carpetas por signo.

2. **Aumento de Datos**

   * A cada secuencia original de keypoints se le aplican transformaciones aleatorias:

     * **Jitter Gaussiano**: se añade ruido a cada coordenada.
     * **Escalado Global**: se multiplica por un factor aleatorio cercano a 1.
     * **Traslación Global**: se desplaza toda la nube de puntos un pequeño porcentaje del rango.
     * **Rotación en Plano XY**: giro alrededor del eje Z.
     * **Dropout de Keypoints**: se fijan aleatoriamente algunos puntos a cero (simula oclusiones).
     * **Warp Temporal**: duplicación o eliminación esporádica de frames para simular cambios de velocidad.
   * De cada muestra original se generan varias versiones augmentadas para enriquecer la diversidad del dataset.

3. **Detección de Ventana Activa**

   * Para cada secuencia augmentada, se calcula la magnitud de movimiento frame a frame, sumando las distancias L2 entre keypoints consecutivos.
   * Se define un umbral proporcional al máximo de velocidad, y se identifican los primeros y últimos frames donde el movimiento supera ese umbral.
   * La región entre esos índices define la “ventana activa” del gesto (evita incluir muchos frames sin movimiento).

4. **Ajuste de Longitud Fija**

   * La ventana activa detectada se recorta o paddea para tener siempre una longitud constante (por ejemplo, 73 frames).
   * Si la duración excede la longitud deseada, se realiza un center-crop. Si dura menos, se añaden frames nulos a ambos lados de manera equitativa.
   * De este modo, todas las secuencias para la LSTM comparten la misma forma y pueden cargarse en batch sin necesidad de padding adicional en el modelado.

5. **Cálculo de Características Auxiliares**

   * La **duración normalizada** se obtiene dividiendo el número de frames de la ventana activa por la longitud fija (e.g., duración/73).
   * La **fracción de manos activas** se calcula separando velocidades left/right, definiendo un umbral de movimiento y midiendo el porcentaje de frames útiles en los que cada mano aparece y realmente se mueve.
   * Estos valores aportan información global sobre el gesto (ej. un gesto largo vs. corto, uso preferente de mano izquierda vs. derecha).

6. **Normalización de Características**

   * Se recopila todo el conjunto de entrenamiento, se concatenan las secuencias `(num_muestras, 73, 126)` (42 keypoints × 3 coordenadas = 126 features) y se calcula la media y desviación estándar por feature.
   * Cada secuencia se normaliza restando la media y dividiendo por la desviación estándar, para estabilizar el entrenamiento de la red y acelerar la convergencia.

---

## Modelos

### 1. LSTM con Características Auxiliares

* **Entrada**: Tensores de forma `(B, 73, 126)` más vectores auxiliares `(B,1)` para duración y `(B,2)` para fracción de uso de cada mano.

* **Arquitectura**:

  1. Capa LSTM (con 2 capas ocultas y tamaño de estado oculto configurable).
  2. Se extrae el último estado oculto (representación fija del gesto).
  3. Se aplica Layer Normalization sobre ese vector.
  4. Se concatenan las características auxiliares (3 valores adicionales).
  5. Dos capas fully-connected: primero con 64 unidades y ReLU, luego capa de salida con tantas neuronas como clases (15).
  6. Softmax final implícito durante la inferencia para convertir logits en probabilidades.

* **Objetivo**: Minimizar entropía cruzada con etiquetas codificadas.

* **Regularización**: Dropout interno en las capas LSTM (si corresponde), y Early-Stopping basado en la pérdida de validación con paciencia configurable.

* **Artefactos Generados**:

  * Pesos del mejor modelo (.pt)
  * LabelEncoder para traducir índices de clase a nombres de signo
  * Vectores `X_mean.npy` y `X_std.npy` para normalizar nuevas muestras en producción

### 2. TCN (Temporal Convolutional Network) con Características Auxiliares

* **Entrada**: Tensores de forma `(B, T, 126)` donde `T` puede variar entre muestras (no es necesario recortar a longitud fija).

* **Arquitectura**:

  1. Capa inicial 1×1 que adapta el número de features (126) a un número deseado de “canales” (p. ej. 64).
  2. Serie de **Bloques Residuales Dilatados**: cada uno realiza:

     * Convolución 1D causal con dilatación predeterminada (se omite “futuro” para mantener causalidad).
     * Recorte de los excesos de padding para preservar la longitud original de la secuencia.
     * ReLU, Dropout, y suma residual (añade la entrada proyectada al resultado).
  3. Tras aplicar todos los bloques, se emplea un **Pooling Global Adaptativo** que colapsa la dimensión temporal variable a longitud 1 (elige el valor máximo a lo largo del tiempo para cada canal).
  4. Se concatenan las características auxiliares (duración normalizada y fracciones de manos, total 3 valores).
  5. Dos capas fully-connected: primero con 64 unidades y ReLU, luego capa de salida con número de clases.

* **Ventaja Principal**:

  * Puede procesar secuencias de longitud variable sin necesidad de recorte/padding previo.
  * Las convoluciones dilatadas permiten abarcar contextos temporales más largos (mirar varios pasos atrás) sin incrementar excesivamente la profundidad.
  * El uso de conexiones residuales facilita el flujo de gradiente y la estabilización de redes profundas.

* **Entrenamiento y Evaluación**:

  * Se usan los mismos conjuntos de entrenamiento y validación que para la LSTM, salvo que las secuencias no tienen longitud fija (TCN aplica pooling global para unificar la dimensión temporal).
  * También se incluyen Early-Stopping y cálculo de métricas de clasificación (classification report y matriz de confusión).

---

## Evaluación y Métricas

Durante y tras el entrenamiento se evalúa el desempeño usando:

* **Accuracy (Exactitud)**
* **Classification Report**: Precision, recall y F1-score por clase.
* **Matriz de Confusión**: Visualización de aciertos y errores por par (clase verdadera, clase predicha).
* **Métricas de Separabilidad** (durante la fase de augmentación):

  * **MSE / RMSE** entre secuencia original y augmentada para cuantificar desviaciones promedio de keypoints.
  * **DTW (Dynamic Time Warping)** para medir distancia entre secuencias con posible desalineación temporal (warp).
  * **SNR (Signal-to-Noise Ratio)** para calibrar nivel de ruido añadido.
  * **Distancia Intra/Inter-Clase**: comparar promedios de distancias (MSE o DTW) dentro de la misma clase versus entre clases, asegurando que intra-clase sea mucho menor que inter-clase.
  * **Silhouette Score** y **Davies–Bouldin Index** para evaluar qué tan bien separadas quedan las clases en el espacio de características tras augmentación.

---

## Despliegue con FastAPI

Para ofrecer una demostración en tiempo real, se ha implementado un servidor FastAPI que:

1. Recibe un vídeo desde el navegador (JavaScript en front-end).
2. Guarda temporalmente el archivo en disco.
3. Extrae los keypoints usando MediaPipe Holistic.
4. Detecta la ventana activa, ajusta la longitud a 73 frames y normaliza.
5. Calcula duración normalizada y fracciones de manos activas.
6. Construye tensores y realiza inferencia con el modelo LSTM entrenado (o TCN si se decide reemplazar).
7. Devuelve la predicción de signo y un puntaje de confianza (probabilidad de la clase máxima).

La comunicación con el front-end se realiza mediante JSON, por ejemplo:

```json
{ "sign": "hello", "confidence": 0.92 }
```

Lo cual permite mostrar la predicción al usuario en una simple aplicación web.

Para levantar la app ejecutar desde la terminal 
```
python3 -m uvicorn app:app --reload    # backend
python3 -m http.server 3000            # frontend
```

---

## Ejecución Paso a Paso

1. **Instalar Dependencias**

   * Asegurarse de contar con Python 3.8+ y los paquetes indicados en `requirements.txt`.

2. **Extraer Keypoints**

   * Ejecutar rutina de extracción para convertir vídeos crudos en archivos `.npy` de keypoints.

3. **Aumentar el Dataset**

   * Aplicar augmentación con los parámetros deseados, generando nuevas muestras.

4. **Preprocesar y Ajustar Longitud**

   * Detectar ventana activa y pad/crop a longitud fija para preparar datos de LSTM. Para TCN, basta con recortar la ventana activa sin forzar longitud.

5. **Entrenar Modelos**

   * En los notebooks (`train_lstm.ipynb` y `train_tcn.ipynb`), seguir el flujo:

     1. Cargar datos con DataLoader (pasan secuencias, duraciones y fracciones de manos).
     2. Normalizar características con vectores `X_mean.npy` y `X_std.npy`.
     3. Inicializar y ajustar LSTM o TCN con Early-Stopping.
     4. Guardar mejores pesos, `label_encoder.pkl`, `X_mean.npy` y `X_std.npy`.

6. **Evaluar Desempeño**

   * Generar el informe de clasificación y matriz de confusión para cuantificar precision, recall, F1-score y exactitud.

7. **Desplegar API**

   * Desde la carpeta `app/`, ejecutar el servidor con Uvicorn; la ruta `/predict` queda disponible para recibir vídeos y devolver predicciones en JSON.

---

## Ventajas y Limitaciones

* **Ventajas**

  * Procesamiento sólo de keypoints de manos (42 puntos × 3 coordenadas) (mas ligero computacionalmente).
  * Aumento de datos robusto que emula ruidos, variaciones de cámara y velocidad de gesto.
  * Dos arquitecturas de modelado: LSTM  y TCN.
  * Incorporación de características globales (duración y uso de manos) que aumenta la capacidad de separación.
  * Pipeline modular y reutilizable, con notebooks bien documentados y API lista para producción.

* **Limitaciones y Futuras Mejoras**

  * Solo se consideran keypoints de manos; no se aprovecha la información del cuerpo o la cara.
  * Actualmente el LSTM requiere longitud fija (pad/crop), lo cual puede introducir información no realista (frames nulos).
  * El TCN maneja longitud variable, pero puede requerir ajuste de hiperparámetros de dilatación según dataset.
  * El modelo podría beneficiarse de arquitecturas que exploten la topología de la mano (e.g., GCNs para grafos esqueléticos).
  * Se podría incluir un front-end más completo (UI) y métricas en tiempo real para mejorar la experiencia final.
  * Se podría mejorar la forma de detectar la ventana activa y el porcentaje de uso de manos para evitar falsos positivos.


