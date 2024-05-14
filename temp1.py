import streamlit as st
import numpy as np
import pandas as pd
import threading
from typing import Union
import cv2
import av
import re
import os
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, ClientSettings, RTCConfiguration,WebRtcMode
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image, ImageColor
from collections import Counter
import numpy as np
import time
import logging
import os
from dotenv import load_dotenv
import streamlit as st
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_path = r"face_shape_classifier.pth"
train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật_Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}

class MyNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Kiểm tra số kênh của tensor
        if tensor.size(0) == 1:  # Nếu là ảnh xám
            # Thêm một kênh để đảm bảo phù hợp với normalize
            tensor = torch.cat([tensor, tensor, tensor], 0)

        # Normalize tensor
        tensor = transforms.functional.normalize(tensor, self.mean, self.std)
        return tensor


# Load lại mô hình đã được huấn luyện
device = torch.device('cpu')  # Sử dụng CPU
model = torchvision.models.efficientnet_b4(pretrained=False)
num_classes = len(train_dataset)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Định nghĩa biến đổi cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    MyNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Load mô hình nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def predict_from_image(image):
    # Chuyển ảnh sang grayscale nếu cần thiết
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Chuyển ảnh sang numpy array
    image_np = np.array(image)

    # Chuyển ảnh sang grayscale để sử dụng mô hình nhận diện khuôn mặt
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Nhận diện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Nếu tìm thấy khuôn mặt, lấy ảnh khuôn mặt và thực hiện dự đoán
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Giả sử chỉ lấy khuôn mặt đầu tiên
        face_img = image.crop((x, y, x + w, y + h))  # Cắt ảnh khuôn mặt từ ảnh gốc

        # Áp dụng biến đổi cho ảnh khuôn mặt
        input_image = transform(face_img).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

        # Thực hiện dự đoán
        with torch.no_grad():
            output = model(input_image)

        # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
        predicted_class_idx = torch.argmax(output).item()

        train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật_Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
        # Lấy tên của nhãn dự đoán từ tập dữ liệu
        predicted_label = train_dataset[predicted_class_idx]

        return predicted_label
    else:
        return "No face detected."


def predict_from_face_image(image):
    
   
    # Áp dụng biến đổi cho ảnh khuôn mặt
    pil_image = Image.fromarray(image)
    input_image = transform(pil_image).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

    # Thực hiện dự đoán
    with torch.no_grad():
        output = model(input_image)

    # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
    predicted_class_idx = torch.argmax(output).item()

    train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật_Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
    # Lấy tên của nhãn dự đoán từ tập dữ liệu
    predicted_label = train_dataset[predicted_class_idx]

    return predicted_label
  
def read_file_content(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()
def main():
    st.title("Face Shape Prediction")

    # Lựa chọn giữa webcam và tải ảnh
    option = st.radio("Choose prediction method:", ("Webcam", "Image"))

    if option == "Webcam":
        # -------------Header Section------------------------------------------------
        # Haar-Cascade Parameters
        
        minimum_neighbors = 3
        # Minimum possible object size
        min_object_size = (50, 50)
        # bounding box thickness
        bbox_thickness = 3
        # bounding box color
        bbox_color = (0, 255, 0)
        with st.sidebar:
    
            title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Nhân Tướng Học Khuôn Mặt</p>'
            st.markdown(title, unsafe_allow_html=True)
            # slider for choosing parameter values
            minimum_neighbors = st.slider("Mininum Neighbors", min_value=0, max_value=10,
                                          help="Tham số xác định số lượng lân cận mà mỗi hình chữ nhật ứng cử viên phải giữ lại. "
                                          "Thông số này sẽ ảnh hưởng đến chất lượng của khuôn mặt được phát hiện. "
                                          "Giá trị cao hơn dẫn đến ít phát hiện hơn nhưng chất lượng cao hơn.",
                                          value=minimum_neighbors)
        
            # slider for choosing parameter values
        
            min_size = st.slider(f"Mininum Object Size, Eg-{min_object_size} pixels ", min_value=3, max_value=500,
                                 help="Kích thước đối tượng tối thiểu có thể. Các đối tượng nhỏ hơn sẽ bị bỏ qua",
                                 value=50)
        
            min_object_size = (min_size, min_size)
        
            # Get bbox color and convert from hex to rgb
            bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00FF00")), "RGB")
        
            # ste bbox thickness
            bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=30,
                                       help="Đặt độ dày của khung giới hạn",
                                       value=bbox_thickness)
        st.markdown(
            "Luư Ý Khi sử dụng:"
            " Bạn hãy mở camera và để app xác định khuôn mặt của bạn. Khi phát hiện ra nó sẽ khoanh vùng khuôn mặt. \n"
            )
        st.warning("NOTE : Nếu khuôn mặt không được phát hiện, bạn có thể thử chụp hình lại nhiều lần")
    
    
        # -------------Sidebar Section------------------------------------------------
        # WEBRTC_CLIENT_SETTINGS = ClientSettings(
        #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        #     media_stream_constraints={"video": True, "audio": False},
        # )
        class VideoTransformer(VideoProcessorBase):
    
            frame_lock: threading.Lock  # transform() is running in another thread, then a lock object is used here for thread-safety.
    
    
            in_image: Union[np.ndarray, None]
    
    
            def __init__(self) -> None:
                self.frame_lock = threading.Lock()
                self.in_image = None
                self.img_list = []
    
    
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                in_image = frame.to_ndarray(format="bgr24")
    
    
                global img_counter
    
                with self.frame_lock:
                    self.in_image = in_image
    
                    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
                    # Draw rectangles around the detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(in_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face = in_image[y:y + h, x:x + w]
                        if len(self.img_list) <=10:
                            self.img_list.append(face)
                return av.VideoFrame.from_ndarray(in_image, format="bgr24")
        img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                          help="Make sure you have given webcam permission to the site")
    
        if img_file_buffer is not None:
    
            with st.spinner("Detecting faces ..."):
                # To read image file buffer as a PIL Image:
                img = Image.open(img_file_buffer)
    
                # To convert PIL Image to numpy array:
                img = np.array(img)
    
                # Load the cascade
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
                # Convert into grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=minimum_neighbors,
                                                      minSize=min_object_size)
                max_area = 0
                max_area_face = None
                if len(faces) == 0:
                    st.warning(
                        "Không phát hiện thấy khuôn mặt nào trong ảnh. Đảm bảo khuôn mặt của bạn được nhìn thấy trong máy ảnh với ánh sáng thích hợp. "
                        "Đồng thời thử điều chỉnh các thông số phát hiện!")
                else:
                    # Draw rectangle around the faces
                    for (x, y, w, h) in faces:
                        # cv2.rectangle(img, (x, y), (x + w, y + h), color=bbox_color, thickness=bbox_thickness)
                        area = w * h
                        if area > max_area:
                            max_area = area
                            max_area_face = (x, y, w, h)
                    # Display the output
                    if max_area_face is not None:
                        # Lấy kích thước và vị trí của khuôn mặt lớn nhất
                        x, y, w, h = max_area_face
                        cv2.rectangle(img, (x, y), (x + w, y + h), color=bbox_color, thickness=bbox_thickness)
                        # Cắt ra hình ảnh của khuôn mặt lớn nhất từ hình ảnh gốc
                        face_img = img[y:y+h, x:x+w]
                        
                        
                    st.image(img)
                    st.image(face_img)
                    if len(faces) > 1:
                        st.success("Total of " + str(
                            len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")
                    else:
                        st.success(
                            "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything")
    
                    
                  
                    if st.button("Predict"):
                       
                        # predicted_label_idx = predict_from_face_image(face_img)
                        predicted_label = predict_from_face_image(face_img)
                        
                        # Chuyển đổi tên nhãn thành tên tệp hợp lệ
                        filename = f"data/{predicted_label}.txt"
                        # sanitized_filename = re.sub(r'[\\/:"*?<>|]+', '_', filename)
                    
                        # Đọc nội dung từ tệp văn bản tương ứng
                        content = read_file_content(filename)
        
                        # Hiển thị nhãn dự đoán
                        st.markdown(
                            f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                            unsafe_allow_html=True)
                    
                        # Hiển thị nội dung từ tệp văn bản
                        st.markdown("**Thông tin chi tiết:**")
                        st.text_area(" ", content, height=300)
                    
                        st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của nhân tướng học ! ♥ ♥ ♥")
                            
    
            hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        elif option == "Image":
            st.write("Upload Image:")
            image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            if image_file is not None:
                image = Image.open(image_file)
                predicted_label = predict_from_image(image)
                if predicted_label != "No face detected.":
                    # Chuyển đổi tên nhãn thành tên tệp hợp lệ
                    filename = f"data/{predicted_label}.txt"
                    # sanitized_filename = re.sub(r'[\\/:"*?<>|]+', '_', filename)
                
                    # Đọc nội dung từ tệp văn bản tương ứng
                    content = read_file_content(filename)
        
                    # Hiển thị nhãn dự đoán
                    st.markdown(
                        f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                        unsafe_allow_html=True)
                
                    # Hiển thị nội dung từ tệp văn bản
                    st.markdown("**Thông tin chi tiết:**")
                    st.text_area(" ", content, height=300)
                    st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của nhân tướng học ! ♥ ♥ ♥")
                else:
                     st.warning(
                            "Không phát hiện thấy khuôn mặt nào trong ảnh. Đảm bảo khuôn mặt của bạn được nhìn thấy trong máy ảnh với ánh sáng thích hợp.\
                            Bạn có thể thử đưa ảnh khác vào để kiểm tra")
                
    
  
if __name__ == "__main__":
    main()
