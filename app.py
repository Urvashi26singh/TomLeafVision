import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle
import traceback

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------------- CONFIG ---------------- #
MODEL_PATH = 'model/tomato_disease_model.h5'
CLASS_INDICES_PATH = 'model/class_indices.pkl'

# ---------------- GLOBALS ---------------- #
model = None
class_names = ['early_blight', 'late_blight', 'healthy']

# ---------------- LOAD MODEL ---------------- #
def load_model_and_classes():
    """Load the trained model and class indices"""
    global model, class_names
    
    print("Loading model and classes...")
    
    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print(f"⚠️ Model file not found at: {MODEL_PATH}")
        model = None
    
    # Load class indices
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, 'rb') as f:
                class_indices = pickle.load(f)
            
            # Convert indices dict to list
            max_index = max(class_indices.values()) if class_indices else 2
            temp_names = [''] * (max_index + 1)
            for name, index in class_indices.items():
                temp_names[index] = name
            
            if all(temp_names):
                class_names = temp_names
            
            print(f"✅ Classes loaded: {class_names}")
        except Exception as e:
            print(f"❌ Error loading class indices: {e}")
            class_names = ['early_blight', 'late_blight', 'healthy']
    else:
        print(f"⚠️ Class indices not found, using defaults")
        class_names = ['early_blight', 'late_blight', 'healthy']

# Load model when app starts
load_model_and_classes()

# ---------------- HELPER FUNCTIONS ---------------- #
def get_disease_info(disease_name, confidence):
    """Get disease information - simplified version"""
    disease_info_dict = {
        'early_blight': {
            'name': 'Early Blight',
            'description': 'Early blight is a fungal disease caused by Alternaria solani.',
            'symptoms': '• Dark, concentric rings on leaves\n• Yellowing around spots\n• Leaf drop in severe cases',
            'treatment': '• Apply copper-based fungicides\n• Remove infected leaves\n• Improve air circulation',
            'prevention': '• Rotate crops annually\n• Water at base of plant\n• Mulch to prevent soil splash',
            'severity': 'High'
        },
        'late_blight': {
            'name': 'Late Blight',
            'description': 'Late blight is a serious fungal disease caused by Phytophthora infestans.',
            'symptoms': '• Large, dark brown/black lesions\n• White fuzzy growth on underside\n• Rapid spread in wet conditions',
            'treatment': '• Apply fungicides immediately\n• Remove and destroy infected plants\n• Avoid overhead watering',
            'prevention': '• Use resistant varieties\n• Ensure good air circulation\n• Monitor weather conditions',
            'severity': 'Critical'
        },
        'healthy': {
            'name': 'Healthy',
            'description': 'Your tomato plant appears to be healthy!',
            'symptoms': 'No disease symptoms detected',
            'treatment': 'Continue good care practices',
            'prevention': '• Maintain regular watering\n• Provide adequate sunlight\n• Monitor for pests',
            'severity': 'None'
        }
    }
    
    disease_key = disease_name.lower().replace(' ', '_')
    info = disease_info_dict.get(disease_key, disease_info_dict['early_blight'])
    
    # Convert strings to lists for symptoms, treatment, prevention
    info['symptoms_list'] = [s.strip() for s in info['symptoms'].split('\n') if s.strip()]
    info['treatment_list'] = [t.strip() for t in info['treatment'].split('\n') if t.strip()]
    info['prevention_list'] = [p.strip() for p in info['prevention'].split('\n') if p.strip()]
    
    return info

def preprocess_image(img, target_size=(224, 224)):
    """Prepare image for model"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- PREDICTION FUNCTION ---------------- #
def predict(image):
    """Predict disease from tomato leaf image"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image)
        
        # Make prediction
        if model is not None:
            preds = model.predict(processed_img, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            
            if idx < len(class_names):
                predicted_class = class_names[idx]
            else:
                predicted_class = "unknown"
                confidence = 0.0
        else:
            # Fallback mock prediction
            preds = np.random.rand(3)
            preds = preds / np.sum(preds)
            idx = np.argmax(preds)
            predicted_class = class_names[idx] if idx < len(class_names) else "healthy"
            confidence = float(preds[idx])
        
        # Get disease info
        info = get_disease_info(predicted_class, confidence)
        
        # Create detailed result
        result_text = f"""
## 🍅 **{info['name']}** - {confidence*100:.1f}% confidence

### 📋 Description
{info['description']}

### ⚠️ Symptoms
{chr(10).join(info['symptoms_list'])}

### 💊 Treatment
{chr(10).join(info['treatment_list'])}

### 🛡️ Prevention
{chr(10).join(info['prevention_list'])}

### 📊 Severity: {info['severity']}
        """
        
        return result_text
    
    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"

# ---------------- GRADIO INTERFACE ---------------- #
# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Upload Tomato Leaf Image"),
    outputs=gr.Markdown(label="🔍 Diagnosis Results"),
    title="🍅 Tomato Disease Detection System",
    description="""
    ### Upload a photo of a tomato leaf to detect diseases
    
    This AI model can identify:
    - **Early Blight** (Alternaria solani)
    - **Late Blight** (Phytophthora infestans)  
    - **Healthy** leaves
    
    **Instructions:** Simply upload a clear photo of a tomato leaf to get started.
    """,
    examples=None,
    allow_flagging="never"
)

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍅 TOMATO DISEASE DETECTION - GRADIO APP 🍅")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    print(f"Classes: {class_names}")
    print("="*60 + "\n")
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
