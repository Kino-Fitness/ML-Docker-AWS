from flask import Flask, request, jsonify, render_template
import os
import tempfile
from PIL import Image
from scripts.measurements import get_predictions
from scripts.kinoscore import get_kino_score
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            if 'frontImage' not in request.files or 'backImage' not in request.files or not request.files['frontImage'] or not request.files['backImage']:
                return jsonify({'error': 'Request is missing required files or files are empty'})
            
            frontImage = request.files['frontImage']
            backImage = request.files['backImage']
            
            if not frontImage.filename.endswith('.jpeg') or not backImage.filename.endswith('.jpeg'):
                return jsonify({'error': 'Images must be in JPEG format'})
            
            weight = request.form.get('weight')
            height = request.form.get('height')
            gender = request.form.get('gender')
            demographic = request.form.get('demographic')

            if not all([frontImage, backImage, weight, height, gender, demographic]):
                return jsonify({'error': 'Missing required parameters'})
            
            if gender not in ['male', 'female']:
                return jsonify({'error': 'Gender must be either "male" or "female"'})
            
            if demographic not in ['white', 'black', 'asian', 'hispanic']:
                return jsonify({'error': 'Demographic must be either "white", "black", "asian", or "hispanic"'})


            temp_dir = 'images'
            os.makedirs(temp_dir, exist_ok=True)

            file_extension1 = os.path.splitext(frontImage.filename)[1]
            file_extension2 = os.path.splitext(backImage.filename)[1]
            
            tempFrontImage = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=file_extension1, delete=False)
            tempBackImage = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=file_extension2, delete=False)
            
            frontImage.save(tempFrontImage.name)
            backImage.save(tempBackImage.name)
            
            frontImage = Image.open(tempFrontImage.name).convert('RGB')
            backImage = Image.open(tempBackImage.name).convert('RGB')
            result = get_predictions(frontImage, backImage, weight, height, gender, demographic)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Method not allowed'})


@app.route('/kino_score', methods=['POST', 'GET'])
def kino_score():
    if request.method == 'POST':
        try:
            user_data = request.get_json()
            result = get_kino_score(user_data)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Method not allowed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
