from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Create Flask app with explicit template folder
app = Flask(__name__, template_folder='templates')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class WebImageClassifier:
    def __init__(self):
        """Initialize the MobileNet model"""
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained MobileNetV2 model"""
        try:
            print("Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet', include_top=True)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, img):
        """Preprocess PIL image for MobileNet"""
        try:
            # Resize image to 224x224
            img = img.resize((224, 224))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to array
            img_array = image.img_to_array(img)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess for MobileNet
            img_array = preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def classify_image(self, img, top_predictions=5):
        """Classify PIL image and return top predictions"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Preprocess the image
        processed_img = self.preprocess_image(img)
        
        if processed_img is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_img)
            
            # Decode predictions to readable labels
            decoded_predictions = decode_predictions(predictions, top=top_predictions)[0]
            
            # Format results
            results = []
            for class_id, label, confidence in decoded_predictions:
                results.append({
                    'label': label.replace('_', ' ').title(),
                    'confidence': float(confidence * 100)
                })
            
            return {"predictions": results}
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

# Initialize classifier
classifier = WebImageClassifier()

@app.route('/')
def home():
    """Serve the main page"""
    try:
        return render_template('interface.html')
    except Exception as e:
        print(f"Template error: {e}")
        return f"<h1>Template Error</h1><p>{str(e)}</p><p>Make sure 'interface.html' is in the 'templates' folder</p>"

@app.route('/classify', methods=['POST'])
def classify():
    """Handle image classification requests"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            # Open image
            img = Image.open(file.stream)
            
            # Classify image
            result = classifier.classify_image(img)
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': classifier.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
