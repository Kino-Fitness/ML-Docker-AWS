import numpy as np
import os
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

model_dir = '/code/files/vbc'
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
weights = [1] * 8
num_folds = 8
OUTPUT_METRICS = ['body_fat', 'muscle_mass', 'bone_mass', 'bone_density']
yolo_model = YOLO("yolov8n.pt")

class MultiInputModel(nn.Module):
    def __init__(self, num_tabular_features, outputs):
        super(MultiInputModel, self).__init__()
        
        # Load pretrained MobileNetV2 and remove the final classification layer
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Batch normalization layer for image features
        self.image_bn = nn.BatchNorm1d(1280, momentum=0.01, eps=1e-3)  # Modified momentum and eps

        # Dense layers for tabular data
        self.tabular_dense1 = nn.Linear(num_tabular_features, 32)
        self.tabular_bn = nn.BatchNorm1d(32, momentum=0.01, eps=1e-3)  # Modified momentum and eps

        # Combined features dimension
        combined_features_dim = 1280 * 2 + 32

        # Define output layers for multiple predictions
        self.output_layers = nn.ModuleList([nn.Linear(combined_features_dim, 1) for _ in outputs])

    def process_image(self, x):
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.image_bn(x)
        return x

    def forward(self, front_image_input, back_image_input, tabular_input):
        # Process images
        front_image_features = self.process_image(front_image_input)
        back_image_features = self.process_image(back_image_input)
        
        # Process tabular data
        tabular_features = F.relu(self.tabular_dense1(tabular_input))
        if tabular_input.size(0) > 1:  # Only apply batch norm if batch size > 1
            tabular_features = self.tabular_bn(tabular_features)
        
        # Combine all features
        combined_features = torch.cat([front_image_features, back_image_features, tabular_features], dim=1)
        
        # Generate outputs for each target
        outputs = [output_layer(combined_features) for output_layer in self.output_layers]
        
        return outputs

def get_vbc(frontImage, backImage, weight, height, gender, demographic, waist, hips):
    X_front_images = np.array([((preprocess_image(frontImage).astype(np.float32) / 255) - mean) / std])
    X_back_images = np.array([((preprocess_image(backImage).astype(np.float32) / 255) - mean) / std])
    X_tabular = np.array([float(height), float(weight), float(waist) / float(hips)])
    return ensemble_predict(X_front_images, X_back_images, X_tabular)


def create_model(num_tabular_features):
    model = MultiInputModel(num_tabular_features, outputs=OUTPUT_METRICS)
    return model

def ensemble_predict(X_front, X_back, X_tabular):
    X_front_tensor = torch.tensor(X_front, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape now [1, 3, 224, 224]
    X_back_tensor = torch.tensor(X_back, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape now [1, 3, 224, 224]
    X_tabular_tensor = torch.tensor(X_tabular, dtype=torch.float32).unsqueeze(0)  # Shape (1, 3)

    preds = []
    for i in range(num_folds):
        model_path = os.path.join(model_dir, f'model_fold_{i}.pt')
        model = create_model(X_tabular.shape[0])
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            preds.append([t.squeeze() for t in model(X_front_tensor, X_back_tensor, X_tabular_tensor)])

        del model

    weighted_preds = []
    for i in range(len(preds[0])):
        weighted_sum = sum(w * lst[i] for w, lst in zip(weights, preds))
        weighted_avg_pred = weighted_sum / sum(weights)
        weighted_preds.append(np.round(weighted_avg_pred.item(), 2))

    response = {
        'body_fat_percentage': weighted_preds[0],
        "muscle_mass": weighted_preds[1],
        "bone_mass": weighted_preds[2],
        "bone_mineral_density": weighted_preds[3]
    }
    
    return response

def preprocess_image(image):
    try:
        image = image.convert('RGB')
        results = yolo_model(image, classes=0)  # or specify custom classes
        boxes = results[0].boxes
        coords = boxes.xyxy.tolist()[0]
        image = image.crop(coords)
        image = image.resize((224, 224))
        image_array = np.array(image)
        return image_array
    
    except:
        raise ValueError("image was not able to be preprocessed by YOLO")