import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_model_safe():
    model_paths = [
        'CNN_model_compatible.keras',
        'CNN_model.keras',
        'brain_tumor_model',
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Attempting to load model from: {model_path}")
                
                if model_path.endswith('.keras'):
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    print("Model recompiled successfully")
                else:
                    model = tf.keras.models.load_model(model_path)
                
                print(f"Model loaded successfully from: {model_path}")
                return model
                
            except Exception as e:
                print(f"Failed to load from {model_path}: {e}")
                continue
    
    try:
        print("Attempting manual model reconstruction...")
        if os.path.exists('model_architecture.json') and os.path.exists('model_weights.h5'):
            from tensorflow.keras.models import model_from_json
            
            with open('model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            
            model = model_from_json(model_json)
            model.load_weights('model_weights.h5')
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model reconstructed and compiled successfully!")
            return model
    except Exception as e:
        print(f"Manual reconstruction failed: {e}")
    
    print("All model loading attempts failed!")
    return None

model = load_model_safe()

if model is not None:
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Model summary:")
    model.summary()
else:
    print("Model failed to load!")

class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def preprocess_image(image):
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    print(f"Original image shape: {img_array.shape}")
    print(f"Original image dtype: {img_array.dtype}")
    print(f"Original image min/max: {img_array.min()}/{img_array.max()}")
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        print("Converted RGBA to RGB")
    elif len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
        print("Converted grayscale to RGB")
    
    img_resized = cv2.resize(img_array, (224, 224))
    print(f"Resized image shape: {img_resized.shape}")
    
    img_float = img_resized.astype(np.float32)
    img_batch = np.expand_dims(img_float, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    print(f"Preprocessed image shape: {img_preprocessed.shape}")
    print(f"Preprocessed image min/max: {img_preprocessed.min():.3f}/{img_preprocessed.max():.3f}")
    print(f"Preprocessed image mean: {img_preprocessed.mean():.3f}")
    
    return img_preprocessed

def classify_brain_tumor(image):
    if model is None:
        return "Error: Model not loaded"
    
    if image is None:
        return "No image provided"
    
    try:
        processed_image = preprocess_image(image)
        
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Image min/max values: {processed_image.min():.3f}/{processed_image.max():.3f}")
        print(f"Image mean: {processed_image.mean():.3f}")
        
        predictions = model.predict(processed_image, verbose=0)
        
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions[0]}")
        
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        print(f"Probabilities: {probabilities}")
        
        predicted_class_index = np.argmax(probabilities)
        predicted_class = class_labels[predicted_class_index]
        
        print(f"Predicted class: {predicted_class}")
        return predicted_class
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return "Processing Error"

def create_interface():
    with gr.Blocks(title="Brain Tumor Classification", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🧠 Brain Tumor Classification
            
            Upload an MRI brain scan image to classify the type of tumor.
            
            **Supported classifications:**
            - Glioma
            - Meningioma  
            - No Tumor
            - Pituitary
            
            *Note: This is for educational/research purposes only and should not be used for medical diagnosis.*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Brain MRI Scan",
                    height=400
                )
                
                classify_btn = gr.Button(
                    "Classify Tumor Type", 
                    variant="primary",
                    size="lg"
                )
                
                gr.Examples(
                    examples=[],
                    inputs=image_input,
                    label="Example Images"
                )
            
            with gr.Column(scale=1):
                prediction_output = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                    lines=2,
                    elem_classes=["prediction-text"]
                )
                
                gr.Markdown(
                    """
                    <style>
                    .prediction-text textarea {
                        font-size: 32px !important;
                        font-weight: bold !important;
                        text-align: center !important;
                        color: #2563eb !important;
                    }
                    </style>
                    
                    ### How to use:
                    1. Upload a brain MRI scan image
                    2. Click "Classify Tumor Type" or the image will be processed automatically
                    3. View the classification result
                    
                    ### About the model:
                    This CNN model was trained to distinguish between different types of brain tumors
                    using MRI scan images.
                    
                    **Disclaimer:** This is for educational purposes only. Do not use for medical diagnosis.
                    """
                )
        
        classify_btn.click(
            fn=classify_brain_tumor,
            inputs=image_input,
            outputs=prediction_output
        )
        
        image_input.change(
            fn=classify_brain_tumor,
            inputs=image_input,
            outputs=prediction_output
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()