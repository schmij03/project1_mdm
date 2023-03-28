from flask import Flask, render_template, request
import onnxruntime
import numpy as np
from PIL import Image
import base64
import urllib.request

app = Flask(__name__)

# Load the model from a URL
model_url = 'https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg19-7.onnx'
model_bytes = urllib.request.urlopen(model_url).read()
session = onnxruntime.InferenceSession(model_bytes)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['image']
    img = Image.open(file.stream)
    img = img.resize((224, 224)) # Resize the image to the required size for VGG-19
    img = np.array(img).transpose(2, 0, 1) # Transpose the image to match ONNX input format
    img = np.expand_dims(img, axis=0) # Add batch dimension
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: img.astype(np.float32)})[0]
    class_idx = np.argmax(pred)
    with urllib.request.urlopen('https://example.com/imagenet_classes.txt') as f:
        classes = f.readlines()
        class_name = classes[class_idx].split(',')[1].strip()
        class_number = classes[class_idx].split(',')[0]
    result = f'Predicted class: {class_name} '

    # Create a new HTML element with the predicted result and uploaded image
    html = f'<div class="alert alert-success" role="alert"> \
                <h4 class="alert-heading">Prediction result_:</h4> \
                <p>{result}</p> \
            </div>'

    return html

if __name__ == '__main__':
    app.run(debug=True)
