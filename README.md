# Diabetic-retinopathy-classification
Combined ResNet50 and InceptionV3 CNN model for image classification

This project presents an automated deep learning-based system for detecting and classifying **Diabetic Retinopathy (DR)** using retinal images. The system leverages a **hybrid CNN model** combining **ResNet50** and **InceptionV3** architectures, trained on the APTOS 2019 dataset. The model is deployed via a web application using Flask, enabling easy use by healthcare providers or patients.

---

## 📘 Introduction

Diabetic Retinopathy (DR) is a severe complication of diabetes that can lead to blindness if left untreated. It damages the retina due to prolonged high blood sugar levels, causing vascular changes. Traditional diagnosis methods involve manual inspection by trained ophthalmologists, which is time-consuming, subjective, and inaccessible in many regions. 

This project offers an automated, scalable, and accurate solution to DR detection using a hybrid CNN model. The model classifies DR into five stages:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

---

## 📊 Dataset

- **Name**: APTOS2019 - 10K Augmented Images
- **Source**: [Kaggle](https://www.kaggle.com/datasets/aitude/aptos-augmented-images)
- Dataset credit: APTOS2019 - 10K Augmented Images by (aitude)Sandeep Kumar

- **Size**: 10,000 high-resolution retinal images
- **Class distribution**: 2000 images per DR severity class

Preprocessing includes resizing, scaling, brightness adjustment, and zooming to ensure uniform quality and reduce overfitting.

---

## 🧠 Model Architecture

A **hybrid model** is developed using:
- **ResNet50** for deep residual learning, effective at capturing complex patterns.
- **InceptionV3** for multi-scale feature extraction.

### Key Steps:
1. Load both pre-trained models without top layers.
2. Extract features from input images.
3. Apply Global Average Pooling to reduce dimensionality.
4. Concatenate the features from both models.
5. Add dense layers, batch normalization, and dropout for generalization.
6. Final classification via softmax (multi-class output).
7. Fine-tune selected layers of both models for dataset-specific learning.

---

## 🏋️ Training Details

- **Loss Function**: Categorical Crossentropy
- **Optimization**: Adam
- **Epochs**: 50
- **Augmentation**: Extensive (flip, rotate, zoom, brightness)
- **Validation**: Used to monitor overfitting
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 📈 Performance Results

- **Accuracy**: ~94%
- **F1-Scores**:
  - Class 0: 0.98
  - Class 1: 0.93
  - Class 2: 0.89
  - Class 3: 0.94
  - Class 4: 0.95
- **Macro/Weighted Average**: 0.94
- **Confusion Matrix**: Strong diagonal presence, minimal misclassification
- **Loss/Accuracy Graphs**: Show strong convergence with minor overfitting

---

## 🌐 Web Application

### Stack Used:
- **Frontend**: HTML, CSS, Bootstrap
- **Backend**: Flask (Python)

### Features:
- Upload a retinal image for prediction
- Display predicted DR stage
- DR stage visual explanations
- Care tips for users
- Fully responsive design for mobile and desktop

### How it works:
1. User uploads a retinal image.
2. Image is preprocessed and passed to the model.
3. The DR stage is predicted and displayed.

## File Structure

kavyasudha2803/

├── MIT License.txt # License file with MIT License text

├── README.md # This file

├── Requirements.txt # Project dependencies

├── resnet50+inceptionv3.ipynb # Jupyter notebook with model code

└── .gitignore # Git ignore rules

---

## 🚀 How to Run the Notebook

1. Open the `.ipynb` notebook in Jupyter, Google Colab, or Kaggle.
2. Upload the APTOS dataset or link it as needed.
3. Run all cells sequentially to train and evaluate the model.

---

## 👥 Authors

- Pagadala Pooja
- Gorrepati Kavya Sudha 
- Akuthota Meghana  
- Vuyyala Likhitha    
- Vemula Teena Mounika  

**Supervisor**: Dr. Habila Basumatary  
*Indian Institute of Information Technology, Pune*

---

## License

This project is licensed under the [MIT License](./MIT%20License.txt).  
See the `MIT License.txt` file for details.

---

## 🔭 Future Scope

- Deploy the model as a web-based application  
- Integrate with patient data for personalized diagnosis  
- Use longitudinal data to predict DR progression  
- Extend to classify other retinal diseases

---

## 🔑 Keywords

Diabetic Retinopathy, Deep Learning, ResNet50, InceptionV3, Hybrid CNN, Medical Imaging, APTOS 2019
