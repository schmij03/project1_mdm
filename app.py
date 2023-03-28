from flask import Flask, render_template, request
import onnxruntime
import numpy as np
from PIL import Image



app = Flask(__name__)

session = onnxruntime.InferenceSession('vgg19-7.onnx')

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
    with open ('imagenet_classes.txt') as f:
        classes = f.readlines()
        class_name = classes[class_idx].split(',')[1].strip()
        result = f'Predicted class: {class_name} '

    # Create a new HTML element with the predicted result and uploaded image
    html = f'<div class="alert alert-success" role="alert"> \
                <h4 class="alert-heading">Prediction result:</h4> \
                <p>{result}</p> \
            </div>'

    return html

if __name__ == '__main__':
    app.run(debug=True)
