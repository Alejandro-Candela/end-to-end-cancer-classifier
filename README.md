# Sistema de Detección Automatizada de Adenocarcinomas Pulmonares

## Descripción del Proyecto

Este trabajo desarrolla un sistema de detección automatizada de adenocarcinomas pulmonares mediante el análisis de imágenes médicas, implementado como una aplicación web. El sistema permite a los usuarios cargar imágenes a través de una interfaz intuitiva, las cuales son procesadas por una API que ejecuta un modelo de clasificación de imágenes basado en técnicas avanzadas de aprendizaje profundo. El modelo proporciona un diagnóstico binario sobre la presencia o ausencia de adenocarcinomas.

Además del objetivo clínico, este proyecto se centra en la creación de una plantilla modular y estructurada que abarca todas las etapas del flujo de trabajo típico en la clasificación de imágenes. La arquitectura propuesta integra componentes como:

- Preprocesamiento de datos
- Modelado
- Despliegue
- Gestión del código

Este enfoque garantiza escalabilidad, mantenibilidad y facilita la replicación del sistema para abordar problemas similares en el ámbito de la visión por computadora.

## Metodología

### 1. Recopilación de Datos

Se utilizó el conjunto de datos de adenocarcinoma pulmonar disponible en Kaggle, compuesto por imágenes categorizadas en dos clases:

- Presencia de adenocarcinomas
- Ausencia de adenocarcinomas

Las imágenes fueron preprocesadas para ajustarse al tamaño requerido por la arquitectura VGG16 (224x224 píxeles) y normalizadas al rango [0, 1].

### 2. Aumento de Datos

Para mejorar la generalización del modelo y evitar el sobreajuste, se implementaron técnicas de aumento de datos:

- Rotaciones de hasta 15 grados
- Escalado aleatorio
- Traslación horizontal y vertical
- Reflejo horizontal

### 3. Modelo Utilizado

Se utilizó la arquitectura VGG16 con pesos preentrenados en ImageNet. Las capas fully connected originales fueron reemplazadas por un clasificador denso que incluye:

- Una capa completamente conectada con 256 unidades y activación ReLU
- Dropout del 50% para reducir el sobreajuste
- Una capa final con activación sigmoide para predicciones binarias

### 4. Entrenamiento del Modelo

El modelo fue entrenado utilizando:

- Función de pérdida: `binary_crossentropy`
- Optimizador: Adam (tasa de aprendizaje inicial: 0.01)
- Épocas: 30
- Tamaño de lote: 16
- División del conjunto de datos: 80% para entrenamiento, 20% para validación

### 5. Evaluación del Modelo

Se evaluó el desempeño mediante las métricas de precisión (accuracy) y pérdida (loss) en el conjunto de validación. También se generaron gráficos para monitorear las curvas de precisión y pérdida durante el entrenamiento.

## Resultados

### Principales Métricas

- **Pérdida (loss):** 0.048
- **Precisión (accuracy):** 99.02%

### Observaciones

- El modelo mostró una alta capacidad para distinguir entre imágenes con y sin adenocarcinomas.
- Se identificaron falsos positivos y negativos principalmente en imágenes de baja resolución o con artefactos, resaltando la importancia de datos de alta calidad.

### Gráficas

Las curvas de precisión y pérdida mostraron una convergencia estable, evidenciando un buen ajuste sin signos significativos de sobreajuste.

## Conclusión

El sistema desarrollado alcanzó una precisión del 99.02% en la detección de adenocarcinomas pulmonares. Además, la plantilla modular diseñada facilita su aplicación a otros problemas de clasificación de imágenes, ofreciendo un enfoque reutilizable y escalable.

### Limitaciones

- Sensibilidad a imágenes de baja resolución
- Presencia de falsos positivos y negativos en ciertos escenarios

### Futuras Líneas de Trabajo

1. Integrar herramientas de explainability como Grad-CAM para interpretar las predicciones del modelo.
2. Evaluar el sistema en un entorno clínico real con datos en tiempo real.
3. Explorar técnicas adicionales como superresolución para mejorar la robustez frente a imágenes de baja calidad.

## Cómo Usar Este Proyecto

1. **Instalar dependencias:**

   ```bash
   conda env create -f environment.yml
   conda activate cnn-cancer-classifier
   ```

2. **Ejecutar pipeline con DVC:**

   ```bash
   dvc init
   dvc repro
   ```

3. **Ejecutar la aplicación web:**

   ```bash
   python app.py
   ```

4. **Abrir la interfaz:**
   Accede a `http://localhost:8080` en tu navegador.

---

**Nota:** Si necesitas más información o tienes alguna pregunta, no dudes en abrir un issue en el repositorio.
