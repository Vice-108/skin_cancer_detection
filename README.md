# Skin Cancer Classification Web Application

This project provides a web-based interface to upload an image and classify the type of skin cancer. The model uses the HAM10000 dataset, which is a large collection of labeled skin lesion images. The system applies machine learning techniques to classify skin lesions and help in early detection of skin cancer.

## Features

- **Web Interface**: Simple and intuitive interface to upload images.
- **Image Classification**: The model classifies skin lesions into different types of skin cancer.
- **HAM10000 Dataset**: The model is trained on the HAM10000 dataset, a benchmark dataset for skin cancer classification.
- **Prediction Results**: Once an image is uploaded, the system returns a prediction along with the type of skin cancer.

## Installation

To run the project locally, follow the instructions below:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification
```

### 2. Set up a Virtual Environment

It’s recommended to set up a virtual environment to manage the dependencies.

```bash
python -m venv venv
source venv/bin/activate   # For Linux/MacOS
venv\Scripts\activate      # For Windows
```

### 3. Install Dependencies

The project uses several Python libraries to function, including `Flask` for the web interface and `TensorFlow` for machine learning. Install them using `pip`:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can manually install the following libraries:

```bash
pip install Flask tensorflow scikit-learn opencv-python matplotlib numpy pillow
```

### 4. Download the HAM10000 Dataset

You can download the dataset from [here](https://www.kaggle.com/datasets/competitions/skin-cancer-classification-ham10000). Extract the images into a directory and place it inside your project folder (e.g., `data/ham10000/`).

### 5. Run the Application

To start the web application locally, run the following command:

```bash
python app.py
```

The web application will be available at `http://127.0.0.1:5000`.

## Usage

1. Open the web application in your browser (`http://127.0.0.1:5000`).
2. Upload an image of a skin lesion through the file upload option.
3. The model will process the image and return a classification result, identifying the type of skin cancer (if present).
4. The possible classifications include melanoma, basal cell carcinoma, squamous cell carcinoma, and benign lesions.

## Model

The classification model is based on deep learning techniques and is trained using the HAM10000 dataset. The model uses convolutional neural networks (CNN) to classify skin lesions. It is designed to provide high accuracy in detecting various types of skin cancer, helping in early diagnosis.

### Model Details

- **Dataset**: HAM10000 (available on Kaggle)
- **Preprocessing**: Image resizing and normalization are applied to prepare the images for the model.
- **Model Architecture**: A Convolutional Neural Network (CNN) was used for classification.
- **Libraries Used**: TensorFlow/Keras for model building and training.

## Folder Structure

```plaintext
skin-cancer-classification/
│
├── app.py                  # Main Flask app for the web interface
├── model.py                # Model definition and loading script
├── requirements.txt        # Python dependencies
├── data/                   # Folder to store dataset
│   └── ham10000/           # Images from the HAM10000 dataset
├── static/                 # Folder for static assets like images, CSS, etc.
├── templates/              # HTML templates for the web interface
└── README.md               # This README file
```

## Contributing

Feel free to fork this project, open issues, and submit pull requests. Contributions are welcome!

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make changes and commit them (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

---

Don't let me know if you need any adjustments or additional information!
