from flask import Flask, request
from flask_cors import CORS 
import base64
from PIL import Image
from io import BytesIO
from keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)
input_shape = (48, 48, 1)  # Assurez-vous que les dimensions correspondent à votre entrée

# Charger le modèle depuis le fichier .h5
cnn_model = load_model('Cnn_Model.h5')

def preprocess_image(image):
    # Prétraiter l'image selon vos besoins
    # Assurez-vous que les dimensions correspondent à votre entrée
    image = image.convert('L').resize((48, 48))
    image_array = np.array(image) / 255.0  # Normalisez les valeurs des pixels entre 0 et 1
    image_array = np.expand_dims(image_array, axis=0)  # Ajoutez une dimension de lot
    return image_array

from flask import jsonify

@app.route('/CNN', methods=['POST'])
def upload_image():
    try:
        image = request.form.get('image')  
        base64_data = image.split(",")[1]  
        image = Image.open(BytesIO(base64.b64decode(base64_data)))
        
        preprocessed_image = preprocess_image(image)
        
        # Faites des prédictions avec le modèle chargé
        predictions = cnn_model.predict(preprocessed_image)
        
        predicted_class = int(np.argmax(predictions))
        predicted_probability = float(predictions[0, predicted_class] * 100)

        # Structurez la réponse JSON
        response_data = {
            "prediction": predicted_class,
            "probability": predicted_probability,
            "success": True
        }

        return jsonify(response_data)
    except Exception as e:
        # En cas d'erreur, retournez une réponse d'erreur
        response_data = {
            "error": str(e),
            "success": False
        }
        return jsonify(response_data), 500  # 500 Internal Server Error


if __name__ == '__main__':
    app.run(debug=True)
