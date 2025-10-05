# 🌌 Proyecto de Detección de Exoplanetas / Exoplanet Detection Project

Este proyecto implementa un sistema de **Machine Learning con Flask** para clasificar posibles exoplanetas usando datos astronómicos de tránsito. El modelo está basado en un **Random Forest** entrenado con el dataset de la NASA Exoplanet Archive.

---

## 🚀 Características / Features

* Clasificación en **3 categorías / 3 categories**:

  * ✅ **Confirmed / Confirmado**
  * ❌ **False Positive / Falso Positivo**
  * 🕒 **Candidate / Candidato**
* Interfaz web desarrollada con **HTML + CSS + JavaScript** / Web interface built with **HTML + CSS + JavaScript**.
* API en **Flask** para entrenamiento y predicción / **Flask API** for training and prediction.
* Entrenamiento automático con dataset CSV / Automatic training with CSV dataset.

---

## 📂 Estructura del Proyecto / Project Structure

```
📦 Proyecto Exoplanetas / Exoplanet Project
 ┣ 📂 excel_splits        # Dataset dividido / Split dataset
 ┣ 📂 imagenes            # Recursos visuales / Visual resources
 ┣ 📜 back.py             # Backend Flask (API ML)
 ┣ 📜 index.html          # Interfaz web / Web interface
 ┣ 📜 styles.css          # Estilos de la interfaz / Interface styles
 ┣ 📜 limpiar_columnas.py # Preprocesamiento del dataset / Dataset preprocessing
 ┗ 📜 README.md           # Documentación / Documentation
```

---

## ⚙️ Instalación y Uso / Installation and Usage

### Español

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/usuario/proyecto-exoplanetas.git
   cd proyecto-exoplanetas
   ```
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el servidor Flask:

   ```bash
   python back.py
   ```
4. Abrir `index.html` en el navegador.

---

### English

1. Clone the repository:

   ```bash
   git clone https://github.com/username/exoplanet-project.git
   cd exoplanet-project
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:

   ```bash
   python back.py
   ```
4. Open `index.html` in your browser.

---

## 📊 Resultados del Modelo / Model Results

* **Precisión (Accuracy):** ~90%
* **F1-Score:** ~84%
* **Datos analizados / Data analyzed:** ~9,200 registros

---

✨ Proyecto académico de clasificación de exoplanetas usando Machine Learning y Flask.
✨ Academic project for exoplanet classification using Machine Learning and Flask.
