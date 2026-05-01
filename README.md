# 🌍 Nature Detection AI (Image Classification System)

## 📌 Project Description

This project is a **Deep Learning-based Image Classification System** that identifies natural scenes from images.

The system is trained to classify images into different environment categories such as:

* 🏔️ Mountains
* 🏢 Buildings
* 🏜️ Desert
* 🌲 Forest
* ❄️ Snow

It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to analyze and predict the class of an input image.

---

## 🚀 Features

* 🧠 Image classification using Deep Learning
* 📊 Trained on real-world dataset (Intel Image Dataset)
* 🔄 Data augmentation for improved accuracy
* 📁 Organized training and testing datasets
* ⚡ Fast prediction results
* 🌐 قابل للتطوير إلى Web أو Mobile App

---

## 🛠 Technologies Used

### Core:

* Python

### AI / Machine Learning:

* TensorFlow
* Keras

### Data Processing:

* NumPy
* ImageDataGenerator (Data Augmentation)

---

## 📁 Project Structure

```
NATURE_DETECTION_PYTHON/
│
├── intel_dataset/
│   ├── seg_train/        # Training images
│   ├── seg_test/         # Testing images
│
├── train.py              # Model training script
├── app_run.py            # Run / prediction script
```

---

## ⚙️ Requirements

Make sure you have the following installed:

* Python 3.8 or higher
* pip (Python package manager)

---

## 📦 Installation

1. Clone the repository:

```
git clone https://github.com/your-username/nature-detection-ai.git
cd nature-detection-ai
```

2. Install dependencies:

```
pip install tensorflow numpy
```

---

## ▶️ How to Run

### 🔹 Step 1: Train the Model

```
python train.py
```

### 🔹 Step 2: Run the Application

```
python app_run.py
```

---

## 🧠 How It Works

1. Images are loaded from dataset folders
2. Data augmentation is applied to improve training
3. CNN model is trained on labeled images
4. Model predicts the category of new images
5. Output shows the predicted class

---

## 📊 Dataset

This project uses the **Intel Image Classification Dataset**, which contains categorized images for:

* Buildings
* Forest
* Glacier (Snow)
* Mountain
* Sea / Desert

---

## 📌 Future Improvements

* 🌐 Build a web interface using Flask
* ⚛️ Integrate with React frontend
* 📱 Convert to mobile app using Expo or Flutter
* 🧠 Use advanced models (ResNet, MobileNet)
* 📷 Add real-time camera detection

---

## ⚠️ Notes

* Training may take time depending on system performance
* GPU is recommended for faster training

---

## 👨‍💻 Author

**Osama Abdullah Obaid**
osamhobaid4@gmail.com
IT Engineer | AI & Web Developer

---

## 📄 License

This project is for educational purposes.
