# web_demo.py
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io
import base64
import os

# Import từ files của bạn
from model import TrafficModel
from data_loader import get_transforms

app = Flask(__name__)

# Config
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES = ['Cam', 'Chidan', 'Hieulenh', 'Nguyhiem', 'Phu']
CLASS_DESCRIPTIONS = {
    'Cam': 'Biển báo cấm - Cấm các hành vi giao thông',
    'Chidan': 'Biển báo chỉ dẫn - Hướng dẫn thông tin',
    'Hieulenh': 'Biển báo hiệu lệnh - Bắt buộc thực hiện',
    'Nguyhiem': 'Biển báo nguy hiểm - Cảnh báo nguy hiểm',
    'Phu': 'Biển báo phụ - Bổ sung thông tin'
}

# Load model và transform từ project của bạn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load transform từ data_loader
_, val_transform = get_transforms()
transform = val_transform  # Dùng transform từ validation

# Load model
model = TrafficModel(num_classes=5)

try:
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print("Please train the model first or check the path.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def predict_image(image):
    """Dự đoán ảnh traffic sign"""
    if model is None:
        return "Model not loaded", 0, []

    try:
        # Preprocess image
        image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        class_name = CLASS_NAMES[predicted.item()]
        confidence_percent = confidence.item() * 100

        # Convert to Python types
        probabilities_list = [float(prob) for prob in probabilities.cpu().numpy()]

        return class_name, confidence_percent, probabilities_list

    except Exception as e:
        return f"Error: {str(e)}", 0, []


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Sign Classification</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            background: rgba(102, 126, 234, 0.05);
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: rgba(102, 126, 234, 0.1);
        }
        .upload-area.dragover {
            border-color: #28a745;
            background: rgba(40, 167, 69, 0.1);
        }
        .result-card {
            transition: all 0.5s ease;
        }
        .prediction-badge {
            font-size: 1.3rem;
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
        }
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            background: #e9ecef;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 1s ease;
        }
        .class-item {
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        .class-item.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="text-center mb-4">
                    <h1 class="text-white display-5 fw-bold mb-2">
                        <i class="fas fa-traffic-light me-2"></i>
                        Traffic Sign Classification
                    </h1>
                    <p class="text-white-50">Upload an image of Vietnamese traffic sign to classify</p>
                </div>

                <div class="glass-card p-4 mb-4">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="upload-area" id="uploadArea">
                                <div id="loadingSpinner" style="display: none;">
                                    <div class="spinner-border text-primary mb-2"></div>
                                    <p>Processing...</p>
                                </div>
                                <div id="uploadContent">
                                    <i class="fas fa-cloud-upload-alt fa-2x text-primary mb-2"></i>
                                    <h5 class="mb-2">Upload Image</h5>
                                    <p class="text-muted mb-2">Drag & drop or click to select</p>
                                    <input type="file" id="fileInput" accept="image/*" class="d-none">
                                    <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
                                        <i class="fas fa-folder-open me-1"></i>Choose Image
                                    </button>
                                </div>
                            </div>

                            <div class="mt-3">
                                <h6 class="mb-2"><i class="fas fa-list-alt me-1"></i>Classes</h6>
                                <div id="classList">
                                    {% for class_name in class_names %}
                                    <div class="class-item">
                                        <i class="fas fa-sign me-1"></i>{{ class_name }}
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>

                        <div class="col-md-6">
                            <div id="resultCard">
                                <div class="text-center mb-3">
                                    <img id="previewImage" class="img-fluid rounded" style="max-height: 150px; display: none;">
                                </div>

                                <div class="text-center mb-3">
                                    <div id="predictionBadge" class="prediction-badge bg-primary text-white d-inline-block mb-2"></div>
                                    <div id="confidenceText" class="text-muted mb-1 small"></div>
                                    <div class="confidence-bar mb-2">
                                        <div id="confidenceFill" class="confidence-fill" style="width: 0%"></div>
                                    </div>
                                    <div id="descriptionText" class="text-muted small"></div>
                                </div>

                                <div class="mt-3">
                                    <h6 class="mb-2"><i class="fas fa-chart-bar me-1"></i>Confidence</h6>
                                    <canvas id="probabilityChart" height="150"></canvas>
                                </div>
                            </div>

                            <div id="placeholder" class="text-center text-muted py-4">
                                <i class="fas fa-image fa-3x mb-2"></i>
                                <p class="mb-0">Upload an image to see results</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="text-center text-white-50 small">
                    <p>Traffic Sign Classification Demo | Powered by PyTorch & Flask</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let probabilityChart = null;
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }

            showLoading();
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('previewImage').src = e.target.result;
                document.getElementById('previewImage').style.display = 'block';
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) showResults(data);
                else showError(data.error);
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error);
            });
        }

        function showLoading() {
            document.getElementById('loadingSpinner').style.display = 'block';
            document.getElementById('uploadContent').style.display = 'none';
        }

        function hideLoading() {
            document.getElementById('loadingSpinner').style.display = 'none';
            document.getElementById('uploadContent').style.display = 'block';
        }

        function showResults(data) {
            document.getElementById('placeholder').style.display = 'none';

            const badge = document.getElementById('predictionBadge');
            badge.textContent = data.predicted_class;
            badge.className = 'prediction-badge bg-success text-white d-inline-block mb-2';

            document.getElementById('confidenceText').textContent = `Confidence: ${data.confidence}%`;
            document.getElementById('confidenceFill').style.width = `${data.confidence}%`;
            document.getElementById('descriptionText').textContent = data.description;

            updateClassHighlighting(data.predicted_class);
            createProbabilityChart(data.all_predictions);
        }

        function updateClassHighlighting(predictedClass) {
            document.querySelectorAll('.class-item').forEach(item => {
                item.classList.toggle('active', item.textContent.includes(predictedClass));
            });
        }

        function createProbabilityChart(predictions) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            if (probabilityChart) probabilityChart.destroy();

            probabilityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: predictions.map(p => p.class),
                    datasets: [{
                        data: predictions.map(p => p.probability),
                        backgroundColor: predictions.map(p => 
                            p.class === predictions[0].class ? 
                            'rgba(40, 167, 69, 0.8)' : 'rgba(102, 126, 234, 0.6)'
                        ),
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        borderRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: { callback: value => value + '%' }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        function showError(message) {
            alert('Error: ' + message);
            document.getElementById('placeholder').style.display = 'block';
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Read and predict
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        class_name, confidence, probabilities = predict_image(image)

        if "Error" in class_name:
            return jsonify({'error': class_name})

        # Convert image for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare results
        class_probs = []
        for i, class_name_item in enumerate(CLASS_NAMES):
            class_probs.append({
                'class': class_name_item,
                'probability': float(round(probabilities[i] * 100, 2)),
                'description': CLASS_DESCRIPTIONS[class_name_item]
            })

        class_probs.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'success': True,
            'predicted_class': class_name,
            'confidence': float(round(confidence, 2)),
            'image_data': f"data:image/png;base64,{img_str}",
            'all_predictions': class_probs,
            'description': CLASS_DESCRIPTIONS[class_name]
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if model else 'not loaded',
        'device': str(device)
    })


if __name__ == '__main__':
    print("🚀 Starting Traffic Sign Classification Demo...")
    print("📊 Available classes:", CLASS_NAMES)
    print("💻 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)