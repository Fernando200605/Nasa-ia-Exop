# 🌌 Proyecto de Detección de Exoplanetas

Este proyecto implementa un sistema de **Machine Learning con Flask** para clasificar posibles exoplanetas usando datos astronómicos de tránsito. El modelo está basado en un **Random Forest** entrenado con el dataset de la NASA Exoplanet Archive.

---

## 🚀 Características

* Clasificación en **3 categorías**:

  * ✅ **Confirmed** (Exoplaneta confirmado)
  * ❌ **False Positive** (Falso positivo)
  * 🕒 **Candidate** (Candidato pendiente de confirmación)
* Interfaz web desarrollada con **HTML + CSS + JavaScript**.
* API en **Flask** para entrenamiento y predicción.
* Entrenamiento automático con dataset CSV.

---

## 📂 Estructura del Proyecto

```
📦 Proyecto Exoplanetas
 ┣ 📂 excel_splits        # Dataset dividido
 ┣ 📂 imagenes            # Recursos visuales
 ┣ 📜 back.py             # Backend Flask (API ML)
 ┣ 📜 index.html          # Interfaz web
 ┣ 📜 styles.css          # Estilos de la interfaz
 ┣ 📜 limpiar_columnas.py # Preprocesamiento de dataset
 ┗ 📜 README.md           # Documentación
```

---

## ⚙️ Instalación y Uso

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

4. Abrir `index.html` en tu navegador.

---

## 📊 Resultados del Modelo

* **Precisión (Accuracy):** ~90%
* **F1-Score:** ~84%

---

## 🌍 English Version

# 🌌 Exoplanet Detection Project

This project implements a **Machine Learning system with Flask** to classify possible exoplanets using astronomical transit data. The model is based on a **Random Forest** trained with the NASA Exoplanet Archive dataset.

---

## 🚀 Features

* Classification into **3 categories**:

  * ✅ **Confirmed** (Confirmed exoplanet)
  * ❌ **False Positive**
  * 🕒 **Candidate** (Pending confirmation)
* Web interface built with **HTML + CSS + JavaScript**.
* **Flask API** for training and prediction.
* Automatic training with CSV dataset.

---

## 📂 Project Structure

```
📦 Exoplanet Project
 ┣ 📂 excel_splits        # Split dataset
 ┣ 📂 imagenes            # Visual resources
 ┣ 📜 back.py             # Flask backend (ML API)
 ┣ 📜 index.html          # Web interface
 ┣ 📜 styles.css          # Interface styles
 ┣ 📜 limpiar_columnas.py # Dataset preprocessing
 ┗ 📜 README.md           # Documentation
```

---

## ⚙️ Installation and Usage

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

## 📊 Model Results

* **Accuracy:** ~90%
* **F1-Score:** ~84%

---

✨ Proyecto académico de clasificación de exoplanetas usando Machine Learning y Flask.
✨ Academic project for exoplanet classification using Machine Learning and Flask.
