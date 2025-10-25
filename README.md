# 🚦 Traffic Sign Recognition Project

A deep learning project for classifying **Vietnamese traffic signs** using **PyTorch**.  
This project includes a complete pipeline from data preprocessing to web deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

This project implements a convolutional neural network (CNN) to classify Vietnamese traffic signs into 5 categories:
- **Cam** - Prohibition signs
- **Chidan** - Direction signs  
- **Hieulenh** - Command signs
- **Nguyhiem** - Danger warning signs
- **Phu** - Supplementary signs

---

## 🎯 Features

- ✅ **Complete ML Pipeline**: Data preprocessing, model training, evaluation, and deployment  
- 🌐 **Web Interface**: User-friendly Flask app for real-time predictions  
- 📊 **Model Monitoring**: TensorBoard integration for training visualization  
- 📈 **Comprehensive Evaluation**: Metrics and confusion matrix analysis  
- 🧩 **Production Ready**: Structured code with error handling and logging  

---

## 📁 Project Structure



traffic_sign_project/
├── data/ # CSV annotations
│ ├── train.csv
│ ├── valid.csv
│ └── test.csv
├── dataset/ # Image datasets
│ ├── train/
│ ├── valid/
│ └── test/
├── models/ # Trained models
│ ├── best_model.pth
│ └── last_model.pth
├── runs/ # TensorBoard logs
├── data_loader.py # Custom dataset and data loading
├── data_preprocessor.py # Data preprocessing utilities
├── model.py # CNN model architecture
├── train.py # Training script
├── evaluate.py # Model evaluation and metrics
├── web_demo.py # Flask web application
├── requirements.txt # Dependencies
└── confusion_matrix.png # Evaluation results


---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition

# Install dependencies
pip install -r requirements.txt

2. Data Preparation

Organize your dataset in the following structure:

dataset/
├── train/
│   ├── Cam/
│   ├── Chidan/
│   ├── Hieulenh/
│   ├── Nguyhiem/
│   └── Phu/
├── valid/
└── test/


Generate CSV annotations:

python data_preprocessor.py

3. Training
# Start training
python train.py

# Resume training from checkpoint
python train.py --resume

# Monitor training with TensorBoard
tensorboard --logdir runs --port 6006


📉 Training & Validation Curves

Training Loss	Validation Accuracy

	
4. Evaluation
# Evaluate model performance
python evaluate.py


📊 Confusion Matrix:

5. Web Demo
# Start the web application
python web_demo.py


Open http://localhost:5000
 in your browser.

🧠 Example Test Results:

Uploaded Image	Model Prediction

	🚫 Cam (Prohibition Sign)

	🟢 Hieulenh (Command Sign)
🏗️ Model Architecture

The project uses a custom CNN architecture:

3 Convolutional Blocks (Conv → BatchNorm → ReLU → Dropout)

Adaptive Average Pooling for flexible input sizes

2 Fully Connected Layers for classification

Optimizer: Adam with Cosine Annealing Scheduler

Loss: Cross Entropy Loss

📊 Performance Metrics
Metric	Value
Overall Accuracy	~95%
Macro F1-score	~0.94
Weighted F1-score	~0.95
Precision	~0.94
Recall	~0.94

Per-class Performance:

Cam: ~96%

Chidan: ~94%

Hieulenh: ~93%

Nguyhiem: ~95%

Phu: ~94%

🌐 Web Interface

The Flask web application provides:

📤 Drag & Drop image upload

⚡ Real-time predictions with confidence scores

📈 Interactive probability charts

💎 Responsive design with glass-morphism UI

📘 Detailed class descriptions

Example:

🛠️ Technical Details
Dependencies

PyTorch 1.9+

TorchVision

Flask

Pillow

Scikit-learn

Pandas

Matplotlib / Seaborn

Data Augmentation

Random Rotation (±10°)

Random Horizontal Flip (30%)

Color Jitter (brightness, contrast, saturation)

Normalization (ImageNet mean/std)

Training Configuration
Parameter	Value
Batch Size	16
Learning Rate	0.001
Epochs	30
Optimizer	Adam
Scheduler	Cosine Annealing
📈 Model Monitoring

The training process is monitored via TensorBoard:

Loss & Accuracy Curves

Learning Rate Scheduling

Model Graph Visualization

Text Summaries per Epoch

Example visualization:

🔧 Customization
Adding New Classes

Update self.classes in data_preprocessor.py

Modify CLASS_NAMES in web_demo.py and evaluate.py

Retrain the model with updated num_classes

Model Architecture

Modify model.py to experiment with:

Additional layers

BatchNorm/Dropout tuning

Custom classifier heads

🤝 Contributing

Contributions are welcome!
Feel free to submit pull requests or open issues for:

🐞 Bug fixes

🚀 Performance improvements

✨ New features

🧾 Documentation updates

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

👨‍💻 Author

Your Name
📂 GitHub: @your-username

📧 Email: your.email@example.com
