import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='captioning_debug.log')

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class CaptioningConfig:
    def __init__(self):
        # Default configurations
        self.max_length = 50
        self.vocab_size = 7578  # Explicitly set to your vocab size
        
        # Load index to word mapping with enhanced error handling
        try:
            with open('index_to_word.json', 'r', encoding='utf-8') as f:
                raw_index_to_word = json.load(f)
            
            # Convert all keys to strings and handle potential integer keys
            self.index_to_word = {}
            for k, v in raw_index_to_word.items():
                # Ensure key is a string
                str_key = str(k)
                self.index_to_word[str_key] = v
            
            logging.info(f"Loaded index_to_word with {len(self.index_to_word)} entries")
            logging.info(f"Sample entries: {list(self.index_to_word.items())[:5]}")
            
            # Verify vocab size matches
            if len(self.index_to_word) != self.vocab_size:
                logging.warning(f"Vocab size mismatch: expected {self.vocab_size}, got {len(self.index_to_word)}")
        
        except FileNotFoundError:
            logging.error("index_to_word.json not found")
            self.index_to_word = {}
        except json.JSONDecodeError:
            logging.error("Could not parse index_to_word.json")
            self.index_to_word = {}

# Initialize configuration
config = CaptioningConfig()

def load_models():
    """Load models with comprehensive error handling"""
    try:
        # Load image feature extractor
        feature_extractor = InceptionV3(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )

        # Load image captioning model
        try:
            captioning_model = tf.keras.models.load_model('model_18.h5', compile=False)
            
            # Log model summary for debugging
            logging.info("Model Summary:")
            captioning_model.summary(print_fn=logging.info)
            
            # Log input and output shapes
            logging.info("Model Input Shapes:")
            for i, input_layer in enumerate(captioning_model.input):
                logging.info(f"Input {i} shape: {input_layer.shape}")
            
            logging.info("Model Output Shape:")
            logging.info(str(captioning_model.output.shape))
            
            # Recompile the model
            captioning_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        except Exception as model_load_error:
            logging.error(f"Error loading model: {model_load_error}")
            return None, None

        return feature_extractor, captioning_model
    
    except Exception as e:
        logging.error(f"Comprehensive model loading error: {e}")
        return None, None

# Global model variables
FEATURE_EXTRACTOR, CAPTIONING_MODEL = load_models()

def preprocess_image(image_path):
    """Enhanced image preprocessing with extensive logging"""
    try:
        # Open and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((299, 299))  # InceptionV3 input size
        
        # Convert to numpy array and preprocess
        image_array = np.array(image)
        image_array = inception_preprocess(image_array)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Extract features
        if FEATURE_EXTRACTOR:
            features = FEATURE_EXTRACTOR.predict(image_array)
            
            logging.info(f"Extracted features shape: {features.shape}")
            
            features_flat = features.flatten()
            
            # Robust padding/truncation to match expected input shape
            if len(features_flat) < 4096:
                padded_features = np.pad(features_flat, (0, 4096 - len(features_flat)), mode='constant')
            else:
                padded_features = features_flat[:4096]
            
            # Reshape to match model's expected input
            padded_features = padded_features.reshape(1, 4096)
            
            logging.info(f"Padded features shape: {padded_features.shape}")
            
            return padded_features
        else:
            logging.error("Feature extractor not loaded")
            raise ValueError("Feature extractor not loaded")
    
    except Exception as e:
        logging.error(f"Detailed image preprocessing error: {e}")
        return None

def prepare_secondary_input(batch_size=1):
    """Prepare secondary input for the model with exact shape matching"""
    try:
        # Determine the correct secondary input shape
        # This might need adjustment based on your model's architecture
        secondary_input = np.zeros((batch_size, 34), dtype=np.float32)
        logging.info(f"Secondary input shape: {secondary_input.shape}")
        return secondary_input
    except Exception as e:
        logging.error(f"Error preparing secondary input: {e}")
        return None

def decode_caption(predictions):
    """Robust caption decoding with comprehensive logging"""
    try:
        # Extensive logging of predictions
        logging.info(f"Predictions shape: {predictions.shape}")
        logging.info(f"Predictions dtype: {predictions.dtype}")
        logging.info(f"Predictions sample: {predictions}")
        
        # Handling multi-dimensional predictions
        if predictions.ndim > 1:
            predicted_indices = np.argmax(predictions, axis=-1)
        else:
            predicted_indices = predictions
        
        logging.info(f"Predicted indices: {predicted_indices}")
        logging.info(f"Predicted indices shape: {predicted_indices.shape}")
        
        # Decode caption with extensive error handling
        caption_words = []
        for idx in predicted_indices.flatten():
            try:
                # Ensure index is converted to string
                str_idx = str(idx)
                
                # Robust word retrieval
                word = config.index_to_word.get(str_idx, f"[UNK:{str_idx}]")
                
                logging.info(f"Index {str_idx} mapped to word: {word}")
                
                # Handle special tokens
                if word == '<end>':
                    break
                if word != '<start>':
                    caption_words.append(word)
            
            except Exception as word_error:
                logging.error(f"Word conversion error for index {idx}: {word_error}")
        
        # Generate final caption
        final_caption = ' '.join(caption_words) if caption_words else "No meaningful caption generated"
        
        logging.info(f"Final caption: {final_caption}")
        return final_caption
    
    except Exception as e:
        logging.error(f"Comprehensive caption decoding error: {e}")
        return f"Caption decoding failed: {str(e)}"

def generate_caption(image_features):
    """Enhanced caption generation with comprehensive debugging"""
    try:
        # Validate model and inputs
        if CAPTIONING_MODEL is None:
            logging.error("Model not loaded correctly")
            return "Model not loaded correctly"
        
        # Validate image features
        if image_features is None:
            logging.error("Invalid image features")
            return "Invalid image features"
        
        # Log input details
        logging.info(f"Image features shape: {image_features.shape}")
        
        # Prepare secondary input
        secondary_input = prepare_secondary_input(batch_size=image_features.shape[0])
        
        if secondary_input is None:
            logging.error("Failed to prepare secondary input")
            return "Failed to prepare secondary input"
        
        logging.info(f"Secondary input shape: {secondary_input.shape}")
        
        # Predict caption with error handling
        try:
            predictions = CAPTIONING_MODEL.predict([image_features, secondary_input])
            logging.info(f"Predictions shape: {predictions.shape}")
            return decode_caption(predictions)
        
        except Exception as predict_error:
            logging.error(f"Prediction process error: {predict_error}")
            return f"Prediction error: {str(predict_error)}"
    
    except Exception as e:
        logging.error(f"Comprehensive caption generation error: {e}")
        return f"Error generating caption: {str(e)}"


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Generate caption with comprehensive error handling
        image_features = preprocess_image(file_path)
        if image_features is not None:
            caption = generate_caption(image_features)
            return render_template('index.html', filename=filename, caption=caption)
        else:
            return render_template('index.html', error="Could not process image")
    
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Additional startup diagnostics
    if FEATURE_EXTRACTOR is None:
        print("WARNING: Feature extractor failed to load")
    if CAPTIONING_MODEL is None:
        print("WARNING: Captioning model failed to load")

    app.run(debug=True)