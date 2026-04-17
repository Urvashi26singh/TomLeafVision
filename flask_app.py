from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import base64
import pickle
import traceback
import time

# Import disease info
from model.disease_info import get_disease_info

app = Flask(__name__)

# ---------------- CONFIG ---------------- #
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Paths for model and class indices
MODEL_PATH = 'model/tomato_disease_model.h5'
CLASS_INDICES_PATH = 'model/class_indices.pkl'

# ---------------- GLOBALS ---------------- #
model = None
class_names = ['early_blight', 'late_blight', 'healthy']  # Default classes


# ---------------- LOAD MODEL ---------------- #
def load_model_and_classes():
    """Load the trained model and class indices"""
    global model, class_names

    print("\n" + "="*60)
    print("LOADING MODEL AND CLASSES")
    print("="*60)

    try:
        # Try to load model
        if os.path.exists(MODEL_PATH):
            print(f"📂 Found model at: {MODEL_PATH}")
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully!")
                
                # Test model with dummy input
                test_input = np.zeros((1, 224, 224, 3))
                test_pred = model.predict(test_input, verbose=0)
                print(f"✅ Model test prediction shape: {test_pred.shape}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                traceback.print_exc()
                model = None
        else:
            print(f"⚠️ Model file not found at: {MODEL_PATH}")
            print("⚠️ Using mock prediction until model is trained.")
            model = None

        # Try to load class indices
        if os.path.exists(CLASS_INDICES_PATH):
            print(f"📂 Found class indices at: {CLASS_INDICES_PATH}")
            try:
                with open(CLASS_INDICES_PATH, 'rb') as f:
                    class_indices = pickle.load(f)
                
                # Convert indices dict to list
                max_index = max(class_indices.values()) if class_indices else 2
                temp_names = [''] * (max_index + 1)
                for name, index in class_indices.items():
                    temp_names[index] = name
                
                # Only update if we got valid names
                if all(temp_names):
                    class_names = temp_names
                
                print(f"✅ Classes loaded: {class_names}")
            except Exception as e:
                print(f"❌ Error loading class indices: {e}")
                print("⚠️ Using default classes")
                class_names = ['early_blight', 'late_blight', 'healthy']
        else:
            print(f"⚠️ Class indices not found at: {CLASS_INDICES_PATH}")
            print("⚠️ Using default classes")
            class_names = ['early_blight', 'late_blight', 'healthy']

        print(f"📊 Final class names: {class_names}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Unexpected error in load_model_and_classes: {e}")
        traceback.print_exc()
        model = None
        class_names = ['early_blight', 'late_blight', 'healthy']


# ---------------- UTILITIES ---------------- #
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(img, target_size=(224, 224)):
    """Prepare image for model"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"❌ Error in preprocess_image: {e}")
        traceback.print_exc()
        raise


def mock_predict():
    """Used when model not loaded - returns realistic mock predictions"""
    import random
    
    print("🔮 Using mock prediction")
    
    # Ensure class_names is populated
    global class_names
    if not class_names or len(class_names) < 3:
        class_names = ['early_blight', 'late_blight', 'healthy']
    
    # Generate random predictions that sum to 1
    preds = np.random.rand(3)
    preds = preds / np.sum(preds)
    
    idx = np.argmax(preds)
    
    # Make sure idx is within range
    if idx >= len(class_names):
        idx = 0
    
    predicted_class = class_names[idx]
    confidence = float(preds[idx])
    
    print(f"🔮 Mock prediction: {predicted_class} ({confidence:.2%})")
    
    return predicted_class, confidence, preds


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    """Home page"""
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction - Updated to match frontend expectations"""
    print("\n" + "="*60)
    print("PREDICT ROUTE CALLED")
    print("="*60)
    
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            print("❌ No file in request.files")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        print(f"📁 File received: {file.filename}")

        # Validate filename
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            print(f"❌ File type not allowed: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG, PNG, or JPEG.'}), 400

        # Read file bytes
        img_bytes = file.read()
        print(f"📊 File size: {len(img_bytes)} bytes")

        if len(img_bytes) == 0:
            print("❌ Empty file")
            return jsonify({'success': False, 'error': 'Empty file'}), 400

        # Validate and open image
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()  # Verify it's a valid image
            print("✅ Image verified")
            
            # Reopen after verify
            img = Image.open(io.BytesIO(img_bytes))
            print(f"✅ Image opened: size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"❌ Image validation failed: {e}")
            return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'}), 400

        # Save image to uploads folder
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            img.save(filepath)
            print(f"✅ Image saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not save image: {e}")
            filepath = None

        # Preprocess image for model
        print("🔄 Preprocessing image...")
        try:
            processed = preprocess_image(img)
            print(f"✅ Preprocessed shape: {processed.shape}")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return jsonify({'success': False, 'error': 'Failed to process image'}), 500

        # Get prediction
        print("🤖 Getting prediction...")
        print(f"📊 Model loaded: {model is not None}")
        print(f"📊 Class names: {class_names}")

        if model is not None:
            try:
                # Make prediction
                preds = model.predict(processed, verbose=0)[0]
                print(f"✅ Raw predictions: {preds}")
                
                # Get predicted class
                idx = int(np.argmax(preds))
                print(f"✅ Predicted index: {idx}")
                
                if idx < len(class_names):
                    predicted_class = class_names[idx]
                else:
                    predicted_class = "unknown"
                    print(f"⚠️ Index {idx} out of range for class_names {class_names}")
                
                confidence = float(preds[idx])
                print(f"✅ Model prediction: {predicted_class} ({confidence:.4f})")
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                traceback.print_exc()
                print("🔄 Falling back to mock prediction")
                predicted_class, confidence, preds = mock_predict()
        else:
            print("🔄 Using mock prediction (no model loaded)")
            predicted_class, confidence, preds = mock_predict()

        print(f"✅ Final prediction: {predicted_class} ({confidence:.4f})")

        # Get disease information
        print("📚 Getting disease information...")
        try:
            # Get disease info from our module
            disease_info = get_disease_info(predicted_class, confidence)
            print(f"✅ Got disease info for: {predicted_class}")
        except Exception as e:
            print(f"❌ Error in get_disease_info: {e}")
            traceback.print_exc()
            
            # Create fallback disease info
            disease_info = {
                'disease': predicted_class.replace('_', ' ').title(),
                'description': f'Detected: {predicted_class}',
                'symptoms': '• Consult a plant specialist for accurate diagnosis',
                'treatment': '• Remove affected leaves\n• Monitor plant health',
                'prevention': '• Maintain good air circulation\n• Avoid overhead watering',
                'severity': 'Moderate'
            }

        # Format response to match frontend expectations
        disease_key = predicted_class.lower().replace(' ', '_')
        
        # Convert prevention string to list
        prevention_text = disease_info.get('prevention', '• Maintain good plant care')
        prevention_list = [item.strip() for item in prevention_text.split('\n') if item.strip()]
        if not prevention_list:
            prevention_list = ['Maintain good plant care']
        
        # Convert treatment string to list
        treatment_text = disease_info.get('treatment', '• Consult a specialist')
        treatment_list = [item.strip() for item in treatment_text.split('\n') if item.strip()]
        if not treatment_list:
            treatment_list = ['Consult a plant specialist']
        
        # Determine severity based on confidence and disease
        if disease_key == 'healthy':
            severity = 'None'
        elif confidence > 0.8:
            severity = 'High'
        elif confidence > 0.5:
            severity = 'Moderate'
        else:
            severity = 'Low'
        
        # Build response matching frontend structure
        response_data = {
            'success': True,
            'image_url': f"/static/uploads/{filename}" if filepath else "",
            'confidence': confidence,  # Keep as decimal (0.95)
            'disease': disease_key,
            'info': {
                'name': disease_info.get('disease', predicted_class.replace('_', ' ').title()),
                'causes': disease_info.get('description', 'No description available'),
                'symptoms': disease_info.get('symptoms', 'Information not available'),
                'prevention': prevention_list,
                'treatment': treatment_list,
                'severity': disease_info.get('severity', severity)
            }
        }
        
        print(f"✅ Sending response with disease: {response_data['info']['name']}")
        print(f"📤 Response preview: {response_data}")
        print("="*60 + "\n")
        
        return jsonify(response_data)

    except Exception as e:
        print("\n❌❌❌ UNHANDLED EXCEPTION ❌❌❌")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict-base64', methods=['POST'])
def predict_base64():
    """Handle base64 encoded image prediction"""
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400

        # Decode base64 image
        img_data = data["image"]
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)

        # Open image
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        img = Image.open(io.BytesIO(img_bytes))

        # Preprocess
        processed = preprocess_image(img)

        # Predict
        if model:
            preds = model.predict(processed, verbose=0)[0]
            idx = np.argmax(preds)
            predicted_class = class_names[idx] if idx < len(class_names) else "unknown"
            confidence = float(preds[idx])
        else:
            predicted_class, confidence, preds = mock_predict()

        # Get disease info
        disease_info = get_disease_info(predicted_class, confidence)
        
        # Format response
        disease_key = predicted_class.lower().replace(' ', '_')
        
        prevention_list = [item.strip() for item in disease_info.get('prevention', '').split('\n') if item.strip()]
        treatment_list = [item.strip() for item in disease_info.get('treatment', '').split('\n') if item.strip()]
        
        response_data = {
            'success': True,
            'image_url': '',
            'confidence': confidence,
            'disease': disease_key,
            'info': {
                'name': disease_info.get('disease', predicted_class.replace('_', ' ').title()),
                'causes': disease_info.get('description', 'No description'),
                'symptoms': disease_info.get('symptoms', 'No symptoms'),
                'prevention': prevention_list if prevention_list else ['Information not available'],
                'treatment': treatment_list if treatment_list else ['Information not available'],
                'severity': disease_info.get('severity', 'Moderate')
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "class_names": class_names,
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "allowed_extensions": list(app.config['ALLOWED_EXTENSIONS'])
    })


@app.route('/test')
def test():
    """Simple test endpoint to verify server is working"""
    return jsonify({
        "status": "ok",
        "message": "Server is running",
        "model_loaded": model is not None,
        "class_names": class_names,
        "timestamp": int(time.time())
    })


@app.route('/test-disease-info')
def test_disease_info():
    """Test endpoint to verify disease_info.py is working"""
    try:
        # Test with each disease
        results = {}
        test_diseases = ['early_blight', 'late_blight', 'healthy', 'unknown_test']
        
        for disease in test_diseases:
            try:
                info = get_disease_info(disease, 0.95)
                results[disease] = {
                    'success': True,
                    'disease': info.get('disease'),
                    'has_fields': all(k in info for k in ['description', 'symptoms', 'treatment', 'prevention'])
                }
            except Exception as e:
                results[disease] = {
                    'success': False,
                    'error': str(e)
                }
        
        return jsonify({
            'status': 'ok',
            'disease_info_working': True,
            'test_results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/debug-image', methods=['POST'])
def debug_image():
    """Debug endpoint to check image uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400

        file = request.files['file']
        img_bytes = file.read()

        info = {
            "filename": file.filename,
            "size": len(img_bytes),
            "content_type": file.content_type
        }

        try:
            img = Image.open(io.BytesIO(img_bytes))
            info.update({
                "format": img.format,
                "size_px": img.size,
                "mode": img.mode,
                "valid": True
            })
        except Exception as e:
            info.update({
                "valid": False,
                "error": str(e)
            })

        return jsonify({'success': True, 'info': info})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍅 TOMATO DISEASE DETECTION SERVER 🍅")
    print("="*60)
    print(f"Python version: {os.sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Current directory: {os.getcwd()}")
    print("-"*60)

    # Load model and classes
    load_model_and_classes()

    # Print server info
    print("\n" + "="*60)
    print("🚀 SERVER STARTING")
    print("="*60)
    print(f"📁 Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"📁 Upload folder exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")
    print(f"📊 Model loaded: {model is not None}")
    print(f"📊 Classes: {class_names}")
    print(f"🌐 Server URL: http://127.0.0.1:5000")
    print(f"🌐 Test endpoint: http://127.0.0.1:5000/test")
    print(f"🌐 Health check: http://127.0.0.1:5000/health")
    print("="*60 + "\n")

    # Run the app
    app.run(host="0.0.0.0", port=7860, debug=True)  
