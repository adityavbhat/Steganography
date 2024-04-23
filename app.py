# app.py
import cv2
from PIL import Image
import numpy as np
from bitstring import BitStream
from flask import Flask, render_template, request, send_file

# Assuming the steganography functions are imported correctly
from steganography import encode_image, decode_image, encode_text, decode_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    if 'message' in request.files:  # Image to image encoding
        carrier_file = request.files['carrier']
        message_file = request.files['message']
        start_pos = int(request.form['start_pos'])
        skip_bits = int(request.form['skip_bits'])

        encoded_image = encode_image(carrier_file, message_file, start_pos, skip_bits)
        encoded_image.save('encoded_image.png')
        return send_file('encoded_image.png', as_attachment=True)
    else:  # Text to image encoding
        carrier_file = request.files['carrier']
        message_text = request.form['message']
        start_pos = int(request.form['start_pos'])
        skip_bits = int(request.form['skip_bits'])

        encoded_image = encode_text(carrier_file, message_text, start_pos, skip_bits)
        encoded_image.save('encoded_image.png')
        return send_file('encoded_image.png', as_attachment=True)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'encoded_file' in request.files:
            encoded_file = request.files['encoded_file']
            # Save the uploaded file to a desired location or perform other operations
            # For now, let's assume we save it with the same name and path
            encoded_file.save('uploaded_file.png')
            return 'File uploaded successfully!'
        else:
            return 'No file uploaded!'
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
