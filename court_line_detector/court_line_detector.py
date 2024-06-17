import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.kps = None

    def resize_with_aspect_ratio(self, image, target_size=224):
        h, w = image.shape[:2]
        if w > h:
            new_w = target_size
            new_h = int(target_size * (h / w))
        else:
            new_h = target_size
            new_w = int(target_size * (w / h))
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_image, new_w, new_h, left, top


    def predict(self, frames):
        # 发现取第0,1张比较不一样
        # 采样几张图片
        len_frames = len(frames)
        # # image_list = []
        # for i, image in enumerate(frames):
        #     if i % int(len_frames//10) == 0 and i > 10:
        #         image_list.append(image)
        # kps_list = []
        image= frames[len_frames//2]
        # for image in image_list:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
            # kps_list.append(keypoints)
        # print(kps_list)
        # kps_array = np.array(kps_list)
        # kps_mean = np.mean(kps_array, axis=0)
        # self.kps = kps_mean
        self.kps = keypoints
        return keypoints
    


    def draw_keypoints(self, image):
        # 画出KPS
        keypoints = self.kps
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames):
        # keypoints = self.kps
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame)
            output_video_frames.append(frame)
        return output_video_frames