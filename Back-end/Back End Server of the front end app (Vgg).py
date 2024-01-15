from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import timm
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms

app = Flask(__name__)
CORS(app)

class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        self.vgg = timm.create_model('vgg16', pretrained=False, num_classes=7)

    def forward(self, images):
        return self.vgg(images)

# Instantiate the model
vgg_model = CustomVGG()

# Load the trained weights
weights_path = 'Vgg_Model.pt'  # Change to the path where your trained weights are stored
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# Rename the keys in the state_dict to match the model's keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("vgg_net", "vgg")
    new_state_dict[name] = v

# Load the modified state_dict into the model
vgg_model.load_state_dict(new_state_dict)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

@app.route('/Vgg', methods=['POST'])
def upload_image():
    try:
        image_data = request.form.get('image')
        base64_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_data)))
        
        preprocessed_image = preprocess_image(image)
        
        vgg_model.eval()
        with torch.no_grad():
            logits = vgg_model(preprocessed_image)

        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits).item()
        predicted_probability = probabilities[0, predicted_class].item() * 100

        response_data = {
            "prediction": predicted_class,
            "probability": predicted_probability,
            "success": True
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    app.run(debug=True)
