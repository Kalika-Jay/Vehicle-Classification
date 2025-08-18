# 🚗 Vehicle Classification

This project is a deep learning-based **vehicle classification system** built using **TensorFlow/Keras**.  
It allows you to train, evaluate, and deploy a model that classifies different types of vehicles from images.  
The project includes a saved model (`.h5`), a training notebook, and a simple application script (`app.py`) for running predictions.

---

## 📂 Project Structure

```

Vehicle-Classification/
│
├── Vehicle Classification.ipynb   # Jupyter notebook for training & evaluation
├── app.py                          # Application script to load model & classify images
├── vehicle\_classification\_model.h5 # Pre-trained Keras model
├── requirements.txt                # Python dependencies
├── data/                           # Dataset (images of vehicles)
├── logs/                           # Training logs & history
└── README.md                       # Project documentation (you are here)

````

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Kalika-Jay/Vehicle-Classification.git
cd Vehicle-Classification
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Training the Model

Open and run the notebook:

```bash
jupyter notebook "Vehicle Classification.ipynb"
```

This will:

* Load the dataset from `data/`
* Preprocess and augment images
* Train the CNN model
* Save the trained model as `vehicle_classification_model.h5`

---

### 2. Running Predictions with `app.py`

The `app.py` script loads the saved model and predicts the class of a given image.

```bash
python app.py --image path_to_image.jpg
```

Example output:

```
Predicted Class: Car
```

---

## 🧠 Model Details

* Framework: **TensorFlow/Keras**
* Type: Convolutional Neural Network (CNN)
* Input size: (check notebook, usually 128x128 or 224x224)
* Output: Predicted vehicle category (Car, Bus, Truck, etc.)

---

## 📊 Logs

Training logs are saved in the `logs/` folder for further analysis and visualization (e.g., with TensorBoard).

---

## ✅ To-Do

* [ ] Add more vehicle categories
* [ ] Improve dataset balance
* [ ] Deploy as a Flask/Django web app
* [ ] Containerize with Docker

