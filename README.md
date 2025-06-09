# Pomegranate-Diseases-Classification-Using-Deep-Learning
An AI-powered system for real-time classification of pomegranate crop diseases using CNNs and weather-based remedies. Achieves 99.9% accuracy with ensemble deep learning. 🌱🧠

A deep learning-based system designed to identify and classify major diseases affecting pomegranate crops. This project integrates **multiple CNN architectures**, **ensemble learning**, and **real-time weather analysis** to enhance prediction accuracy and generate intelligent remedies, aiding in sustainable and precision agriculture.

> **Submitted to:** ICRTAIDS-2025 - *International Conference on Recent Trends in Artificial Intelligence and Data Science*
> **Academic Year:** 2024-2025
> **Institution:** Ajeenkya D. Y. Patil School of Engineering, Pune

---

## ✨ Overview

Pomegranate farming is under threat from diseases such as:

* Alternaria
* Anthracnose
* Bacterial Blight
* Cercospora

These diseases can result in **up to 90% crop loss** under favorable conditions. Traditional detection methods are manual, slow, and require expert knowledge. This project proposes an **automated disease classification system** using:

* Deep Convolutional Neural Networks (CNN)
* Transfer learning (VGG16, ResNet50, MobileNetV2)
* Ensemble prediction for increased accuracy
* Weather-based remedy suggestions using OpenWeatherMap API
* Interactive web app using Streamlit

---

## 🎯 Objectives

* Accurately classify pomegranate leaf diseases
* Build ensemble models to boost reliability
* Integrate weather data for real-time, context-aware recommendations
* Provide a user-friendly system for farmers and agri-tech experts

---

## 📊 Results & Performance

| Model        | Accuracy  | Precision | Recall | F1-Score |
| ------------ | --------- | --------- | ------ | -------- |
| Custom CNN   | 99.4%     | 0.99      | 0.99   | 0.99     |
| VGG16        | 99.6%     | \~0.99    | \~0.99 | \~0.99   |
| ResNet50     | 99.8%     | 1.00      | 1.00   | 1.00     |
| MobileNetV2  | 98.8%     | \~0.99    | \~0.99 | \~0.99   |
| **Ensemble** | **99.9%** | \~1.00    | \~1.00 | \~1.00   |

---

## 🌐 Architecture

```text
User Input
 └── Upload image + location
     ├── Image preprocessing (resize, normalize)
     ├── Weather data fetch (OpenWeatherMap API)
     ├── Prediction from 4 CNN models
     ├── Ensemble prediction (mean average)
     └── Disease label + advisory
```

---

## 📚 Key Features

* Real-time disease detection
* Ensemble model for enhanced precision
* Recommends fungicides based on humidity
* Streamlit web interface
* Responsive on desktop (working on mobile devices)

---

## 🧰 Tech Stack

* **Programming Language:** Python 3.9+
* **Frameworks/Libraries:**

  * TensorFlow, Keras
  * NumPy, Pandas
  * Streamlit
  * Matplotlib, Seaborn
* **API:** OpenWeatherMap
* **Platform:** Jupyter Notebook, VS Code

---

## 🖥️ System Requirements

* Python 3.9+
* 8 GB RAM (16 GB recommended)
* GPU (Optional but improves performance)
* Internet connection (for weather data)

---

## 📁 Folder Structure

```bash
pomegranate-disease-classifier/
├── dataset/               # Images of Fruit (healthy and diseased)
├── models/                # Pre-trained model weights (H5 files)
├── app.py                 # Streamlit web application
├── weather_utils.py       # Fetches live weather data
├── preprocess.py          # Preprocessing functions
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
```

---


## 📊 Dataset

The project uses the [Pomegranate Fruit Diseases Dataset](https://www.kaggle.com/datasets/sujaykapadnis/pomegranate-fruit-diseases-dataset) by Sujay Kapadnis, hosted on Kaggle. It contains labeled images of healthy and diseased pomegranate fruits across multiple classes.

### 📥 How to Download

You can download the dataset manually or using the Kaggle API:

#### 📌 Option 1: Manual Download

1. Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/sujaykapadnis/pomegranate-fruit-diseases-dataset)
2. Sign in and click **Download**.
3. Extract the dataset into the `dataset/` folder of this project.

#### 🐍 Option 2: Using the Kaggle API

If you have the Kaggle CLI installed and configured:

```bash
kaggle datasets download -d sujaykapadnis/pomegranate-fruit-diseases-dataset
unzip pomegranate-fruit-diseases-dataset.zip -d dataset/
```

Make sure the final structure looks like this:

```bash
pomegranate-disease-classifier/
├── dataset/
│   ├── Alternaria/
│   ├── Anthracnose/
│   ├── Bacterial Blight/
│   └── Cercospora/
```

## ⚙️ Installation & Running the Project

### Step 1: Clone the Repository

```bash

cd pomegranate-disease-classifier
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Your OpenWeatherMap API Key

Edit `weather_utils.py` and add your API key:

```python
API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
```

### Step 5: Run the Streamlit App

```bash
streamlit run app.py
```

### Step 6: Using the App

1. Upload a pomegranate fruit image.
2. Enter your location (city).
3. View the predicted disease and the remedy advisory.

---

## ⛅ Weather-Aware Remedies

Our system recommends remedies **only when** certain environmental thresholds are met. Example:

* If **humidity > 70%** and disease = Anthracnose → Suggest fungicide
* Custom recommendations provided via integrated API + rule-based logic

---

## 🎨 Screenshots



---

## 📄 Citation

If you use this work in your research, cite us as:

```bibtex
@inproceedings{thawal2025pomegranate,
  title={Pomegranate Disease Classification Using Deep Learning},
  author={Omkar Thawal and Mayurbhai Chaudhari and Avinash Salunke},
  booktitle={International Conference on Recent Trends in Artificial Intelligence and Data Science (ICRTAIDS)},
  year={2025}
}
```

---

## ✉️ Contact

For academic or research queries:

* Omkar Thawal – [omkar.thawal@dypic.com](mailto:omkarthawal555@gmail.com)

---

## 💬 Acknowledgements

* Project Guide: **Prof. Pooja Dehankar**
* Department of AI & Data Science, ADYPSOE

---

## 📜 License

© 2025 Omkar Thawal, Mayurbhai Chaudhari, Avinash Salunke
**Ajeenkya D. Y. Patil School of Engineering, Pune**
All rights reserved.

This work is submitted under academic evaluation and **ICRTAIDS-2025 conference proceedings**. Unauthorized use, copying, or distribution of this codebase, research, or documentation without written permission is strictly prohibited.

---
