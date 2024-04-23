import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import transforms
from collections import Counter
import numpy as np
import time

# Load lại mô hình đã được huấn luyện
model_path = r"face_shape_classifier.pth"
train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài',
                 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}


class MyNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if tensor.size(0) == 1:  # Nếu là ảnh xám
            tensor = torch.cat([tensor, tensor, tensor], 0)

        tensor = transforms.functional.normalize(tensor, self.mean, self.std)
        return tensor


# Load lại mô hình đã được huấn luyện
device = torch.device('cpu')
model = torchvision.models.efficientnet_b4(pretrained=False)
num_classes = len(train_dataset)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    MyNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict_from_list(images):
  try:
    predicted_labels = []
    for image in images:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        try:
            gray = Image.fromarray(gray)
            input_image = transform(gray).unsqueeze(0)

            with torch.no_grad():
                output = model(input_image)

            predicted_class_idx = torch.argmax(output).item()

            predicted_label = train_dataset[predicted_class_idx]
            predicted_labels.append(predicted_label)
        except:
            predicted_labels.append("No face detected")

    label_counts = Counter(predicted_labels)
    most_common_label = label_counts.most_common(1)[0][0]
  

    return most_common_label
  except:
    pass


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.frame_list = []

    def transform(self, frame):
        if len(self.frame_list) >= 25:
            self.consumer.stop()

        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y:y + h, x:x + w]
            self.frame_list.append(face)

        if len(self.frame_list) >= 10:
            st.stop()

        return img


class_info = {
    'Khuôn mặt trái xoan': {
        'description': '...',
        'careers': ['Truyền thông và Quảng cáo', 'Nghệ thuật và Văn hóa', 'Giáo dục và Đào tạo']
    },
    'Khuôn mặt trái tim': {
        'description': '...',
        'careers': ['Kinh doanh và Quản lý', 'Nghệ thuật và Sáng tạo']
    },
    'Khuôn mặt hình chữ nhật/Khuôn mặt dài': {
        'description': '...',
        'careers': ['Luật sư và Pháp luật', 'Nghiên cứu và Phát triển', 'Tài chính và Đầu tư']
    },
    'Khuôn mặt tròn': {
        'description': '...',
        'careers': ['Y tế và Chăm sóc sức khỏe', 'Tình nguyện và Cứu trợ', 'Giáo dục và Đào tạo']
    },
    'Khuôn mặt vuông': {
        'description': '...',
        'careers': ['Xây dựng và Bất động sản', 'Thể thao và Thể dục', 'Kinh doanh và Quản lý']
    },
    'No face detected': {
        'description': '',
        'careers': ['']
    },
  'None': {
        'description': '',
        'careers': ['']
    }
}


def main():
    st.title("Face Shape Prediction")

    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # if webrtc_ctx.video_transformer:
    #     predicted_label=None
    #     while predicted_label is None:
    #       predicted_label = predict_from_list(webrtc_ctx.video_transformer.frame_list)

    #     st.subheader("Hình Dạng Khuôn mặt:")
    #     st.markdown(
    #         f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
    #         unsafe_allow_html=True)

    #     st.markdown('**Ngành Nghề Phù Hợp:**')
    #     for career in class_info[predicted_label]['careers']:
    #         st.markdown(f"- {career}")
    #     st.markdown('**Đặc điểm tính cách:**')
    #     st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")


if __name__ == "__main__":
    main()
