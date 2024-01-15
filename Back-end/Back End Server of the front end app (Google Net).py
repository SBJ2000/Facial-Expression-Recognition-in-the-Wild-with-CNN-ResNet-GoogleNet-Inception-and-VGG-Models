from flask import Flask, request, jsonify
from flask_cors import CORS 
import base64
from PIL import Image
from io import BytesIO
import torch
from torch import nn 
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models

app = Flask(__name__)
CORS(app)

class CustomGoogleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomGoogleNet, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)

# Instantiate the GoogleNet model
googlenet_model = CustomGoogleNet(num_classes=7)

# Load the trained weights
weights_path = 'GoogleNet_Model.pt'
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
googlenet_model.load_state_dict(state_dict)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

@app.route('/GoogleNet', methods=['POST'])
def upload_image():
    try:
        # Access form data from the request
        image = request.form.get('image')  
        base64_data = image.split(",")[1]  
        image = Image.open(BytesIO(base64.b64decode(base64_data)))
        
        preprocessed_image = preprocess_image(image)
        
        # Make predictions with the loaded model
        googlenet_model.eval()
        with torch.no_grad():
            logits = googlenet_model(preprocessed_image)

        probabilities = F.softmax(logits, dim=1)

        # Process logits as needed
        predicted_class = torch.argmax(logits).item()
        predicted_probability = probabilities[0, predicted_class].item() * 100 

        # Return prediction results or any relevant information
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
