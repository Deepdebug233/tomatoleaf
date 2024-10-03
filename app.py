import base64
from flask import Flask, render_template, request, redirect
import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the models
leaf_model = load_model('res.h5')  # Adjust the model name as needed
disease_model_1 = load_model('tomato_disease_classification.h5')
disease_model_2 = load_model('vggtomato_disease_model.h5')

# Preprocessing function
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        raise ValueError(f"Unsupported image mode: {image.mode}. Please provide an RGB or RGBA image.")
    image = image.resize((224, 224))  # Resize image as required by the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload or video capture
        if 'file' in request.files and request.files['file'].filename != '':
            # Handle file upload
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read()))
        elif 'captured_image' in request.form and request.form['captured_image'] != '':
            # Handle video capture
            captured_image_data = request.form['captured_image']
            image_data = base64.b64decode(captured_image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            return redirect(request.url)

        # Process the image
        processed_image = preprocess_image(image)
        
        # Check if the image is a leaf
        leaf_prediction = leaf_model.predict(processed_image)

        if leaf_prediction[0][0] > 0.9:  # Adjust threshold if needed
            result = "Not a leaf"
        else:
            result = "Leaf"

            # Use the selected model for disease prediction
            selected_model = request.form.get('model')
            if selected_model == 'disease_model_1':
                disease_prediction = disease_model_1.predict(processed_image)
            else:
                disease_prediction = disease_model_2.predict(processed_image)

            disease_class = np.argmax(disease_prediction, axis=1)
            
            # Mapping of class indices to disease names
            disease_labels = [
                'Bacterial spot', 'Early blight', 'Late blight', 
                'Leaf mold', 'Septoria leaf spot', 'Spider mites', 
                'Target spot', 'Yellow leaf curl virus', 
                'Mosaic virus', 'Healthy'
            ]
            disease_result = disease_labels[disease_class[0]]
            
            result = f"Leaf with {disease_result}"

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
