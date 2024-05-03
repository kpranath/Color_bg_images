from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms.functional as T
from datetime import datetime
import os
import cv2

app = Flask(__name__)

#define model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ColorizationResNet(nn.Module):
    def __init__(self):
        super(ColorizationResNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)

        # Stacking Residual Blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Upsampling Layers
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((150, 150)),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150, 150)),
])

checkpoint = torch.load('keshavap_ss675_neemageo_resnet.h5', map_location=torch.device('cpu'))
model = ColorizationResNet()

model.load_state_dict(checkpoint)

model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2LAB)
    bw_lchannel, bw_achannel, bw_bchannel = cv2.split(image_file)
    input_tensor = transform(bw_lchannel)

    with torch.no_grad():
        prediction = model(input_tensor)

    _, predicted = torch.max(prediction.data, 1)
    predicted_np = predicted.detach().cpu().numpy()
    predicted_ab_tensor = predicted_np.tolist()

    pred_a_tensor = pred_ab_tensor[0]
    pred_b_tensor = pred_ab_tensor[1]

    L_image = T.to_pil_image(input_tensor)
    pred_A_image = T.to_pil_image(pred_a_tensor)
    pred_B_image = T.to_pil_image(pred_b_tensor)

    image_size = (150, 150)
    L_image = T.resize(L_image, image_size)
    pred_A_image = T.resize(pred_A_image, image_size)
    pred_B_image = T.resize(pred_B_image, image_size)

    pred_LAB_image = Image.merge("LAB", (L_image, pred_A_image, pred_B_image))
    predicted_image = pred_LAB_image.convert("RGB")

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_predicted.png"
    filepath = os.path.join('./', filename)
    pred_rgb_image.save(filepath)

    return jsonify({'message': 'Image processed and saved successfully', 'filename': filename})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)