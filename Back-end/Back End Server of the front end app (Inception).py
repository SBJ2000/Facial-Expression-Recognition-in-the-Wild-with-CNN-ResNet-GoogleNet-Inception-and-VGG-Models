from flask import Flask, request, jsonify
from flask_cors import CORS 
import base64
from PIL import Image
from io import BytesIO
import timm 
import torch
from torch import nn 
from torchvision import transforms
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()
        # Choisissez l'architecture Inception souhaitée
        self.inception_net = timm.create_model('inception_v3', pretrained=True, num_classes=7)
    
    def forward(self, images, labels=None):
        logits = self.inception_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits

model = InceptionModel()

# Chargez les poids depuis un fichier
weights_path = 'Inception_Model.pt'
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Convertir en mode 'RGB' pour supprimer le canal alpha
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Ajoutez une dimension de lot
    return image




@app.route('/Inception', methods=['POST'])
def upload_image():
    try:
        # Accédez aux données du formulaire depuis la demande
        image = request.form.get('image')  
        base64_data = image.split(",")[1]  
        image = Image.open(BytesIO(base64.b64decode(base64_data)))
        
        preprocessed_image = preprocess_image(image)
        
        # Faites des prédictions avec le modèle chargé
        model.eval()
        with torch.no_grad():
            logits = model(preprocessed_image)

        probabilities = F.softmax(logits, dim=1)

        # Traitez les logits selon vos besoins
        predicted_class = torch.argmax(logits).item()
        predicted_probability = probabilities[0, predicted_class].item() * 100 

        # Retournez les résultats de la prédiction ou toute information pertinente
        response_data = {
            "prediction": predicted_class,
            "probability": predicted_probability,
            "success": True
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500  # 500 Internal Server Error

if __name__ == '__main__':
    app.run(debug=True)
