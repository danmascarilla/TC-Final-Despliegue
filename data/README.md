# ✈️ Predicción de Satisfacción de Pasajeros - Airline Analytics

![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completado-success?style=for-the-badge)

## Componentes del Proyecto

* [jonatan-luzon](https://github.com/jonatan-luzon)
* [IvanMontero04](https://github.com/IvanMontero04)
* [danmascarilla](https://github.com/danmascarilla)

## 📖 1. Objetivo del Proyecto
Este proyecto de Machine Learning tiene como objetivo clasificar y predecir el nivel de **satisfacción de los pasajeros** de una aerolínea basándose en su experiencia de vuelo. 

El propósito de negocio es proporcionar a la compañía una herramienta diagnóstica que identifique qué servicios específicos (WiFi, confort, limpieza, etc.) tienen un impacto directo en la fidelización del cliente, permitiendo optimizar la inversión en calidad de servicio.

---

## 📊 2. El Dataset
Se ha utilizado un dataset con más de **120.000 registros** que incluye:
- **Variables Demográficas:** Edad, Género, Tipo de Cliente (Loyal/disloyal).
- **Variables de Vuelo:** Tipo de viaje (Business/Personal), Clase (Business/Eco), Distancia.
- **Servicios (Rating 1-5):** WiFi, Comodidad de asiento, Entretenimiento, Comida, Limpieza, etc.
- **Retrasos:** Minutos de retraso en salida y llegada.



---

## 🛠️ 3. Preprocesamiento y Feature Engineering
Para cumplir con los estándares profesionales y evitar el **Data Leakage**, se ha implementado un **Pipeline de Scikit-Learn**. Las decisiones técnicas tomadas son:

1.  **Imputación de Nulos:** La variable `Arrival Delay in Minutes` presentaba valores faltantes, los cuales fueron imputados con la **mediana** para no ser afectados por valores extremos (outliers).
2.  **Transformación Categórica:** Uso de `OneHotEncoder` para convertir variables de texto (Gender, Class, etc.) en vectores numéricos procesables por el modelo.
3.  **Escalado Numérico:** Aplicación de `StandardScaler` para normalizar las distancias y edades, asegurando que todas las características tengan el mismo peso inicial.
4.  **Flujo Atómico:** Todo el preprocesamiento está encapsulado junto al modelo final, permitiendo que el set de **Test** se transforme de manera idéntica al de **Train**.



---

## 🤖 4. Modelado y Evaluación
Se seleccionó el algoritmo **Random Forest Classifier** tras comparar diferentes modelos, debido a su robustez y capacidad para capturar relaciones no lineales entre los servicios y la satisfacción.

### Rendimiento en el Set de Test (Datos no vistos):
| Métrica | Resultado |
| :--- | :--- |
| **Accuracy** | **~96%** |
| **F1-Score** | **0.95** |
| **Precisión** | **0.97** |

### Análisis de Importancia de Variables:
El modelo determinó que los factores críticos para la satisfacción son:
1.  **Online Boarding** (Embarque online).
2.  **Inflight WiFi Service**.
3.  **Type of Travel** (Los viajeros de negocios son más propensos a la satisfacción si el servicio es eficiente).



---

## 📂 5. Estructura del Repositorio
```text
├── data/
│   ├── train.csv            # Datos de entrenamiento
│   └── test.csv             # Datos de evaluación final
├── src/
│   └── models/
│       └── model_final_airline.joblib  # Pipeline + Modelo exportado
├── notebooks/
│   └── test.ipynb           # Notebook principal con EDA y Modelado
└── README.md                # Documentación del proyecto
```

### 📻 6. Enlace al video

https://drive.google.com/file/d/1ac4XR-lftOEXUixvRufOensnFLW99cd4/view?usp=drive_link
