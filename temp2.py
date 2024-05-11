import streamlit as st
import numpy as np
import pandas as pd
import threading
from typing import Union
import cv2
import av
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

class_info = {
    'Khuôn mặt trái xoan': {
        'description': 'Những người có khuôn mặt hình trái xoan không bao giờ sai lời nói. Họ luôn biết dùng từ ngữ phù hợp trong mọi tình huống – nghiêm túc hay vui vẻ. Mọi người tôn trọng họ về cách ăn nói và họ cũng có thể hòa hợp với các nhóm tuổi khác nhau nhờ kỹ năng giao tiếp hiệu quả. Đôi khi họ có thể quá tập trung vào việc nói tất cả những điều đúng đắn, điều này có thể khiến họ mất đi những cuộc trò chuyện không được lọc và những khoảnh khắc gắn kết',
        'careers': ['Truyền thông và Quảng cáo', 'Nghệ thuật và Văn hóa', 'Giáo dục và Đào tạo']
    },
    'Khuôn mặt trái tim': {
        'description': 'Những người có khuôn mặt hình trái tim là người có tinh thần mạnh mẽ. Đôi khi họ có thể quá bướng bỉnh, chỉ muốn mọi việc được thực hiện theo một cách cụ thể. Về mặt tích cực, họ lắng nghe trực giác của mình, điều này bảo vệ họ khỏi rơi vào những tình huống nguy hiểm. Họ cũng rất sáng tạo trong bất cứ điều gì họ làm.',
        'careers': ['Kinh doanh và Quản lý', 'Nghệ thuật và Sáng tạo']
    },
    'Khuôn mặt hình chữ nhật/Khuôn mặt dài': {
        'description': 'Bạn đã bao giờ nghe nói về việc đọc khuôn mặt và lòng bàn tay chưa? Vâng, ngay cả hình dạng khuôn mặt cũng có thể tiết lộ rất nhiều điều về tính cách của bạn. Nếu bạn có khuôn mặt hình chữ nhật, bạn tin tưởng nhiều vào suy nghĩ. Bạn dành thời gian suy nghĩ trước khi đưa ra bất kỳ quyết định quan trọng nào. Kết quả là bạn có thể suy nghĩ quá nhiều.',
        'careers': ['Luật sư và Pháp luật', 'Nghiên cứu và Phát triển', 'Tài chính và Đầu tư']
    },
    'Khuôn mặt tròn': {
        'description': 'Những người có khuôn mặt tròn là những người có trái tim nhân hậu. Họ tin vào việc giúp đỡ người khác và làm từ thiện. Do có tấm lòng bao dung nên đôi khi họ không ưu tiên bản thân mình, điều này có thể dẫn đến những kết quả không mấy tốt đẹp cho bản thân họ',
        'careers': ['Y tế và Chăm sóc sức khỏe', 'Tình nguyện và Cứu trợ', 'Tình nguyện và Cứu trợ']
    },
    'Khuôn mặt vuông': {
        'description': 'Những người có khuôn mặt này thường khá mạnh mẽ - cả về thể chất cũng như tình cảm. Tuy nhiên, hãy đảm bảo rằng bạn tiếp tục nuôi dưỡng những điểm mạnh của mình, nếu không chúng sẽ chỉ ở mức bề nổi trong tương lai.',
        'careers': ['Xây dựng và Bất động sản', 'Thể thao và Thể dục', 'Kinh doanh và Quản lý']
    },
    'No face detected': {
        'description': '',
        'careers': ['']
    }
}
model_path = r"face_shape_classifier.pth"
train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}

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


def predict_from_face_image(image):
    
   
    # Áp dụng biến đổi cho ảnh khuôn mặt
    pil_image = Image.fromarray(image)
    input_image = transform(pil_image).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

    # Thực hiện dự đoán
    with torch.no_grad():
        output = model(input_image)

    # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
    predicted_class_idx = torch.argmax(output).item()

    train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
    # Lấy tên của nhãn dự đoán từ tập dữ liệu
    predicted_label = train_dataset[predicted_class_idx]

    return predicted_label
  

# Định nghĩa hàm dự đoán qua ảnh
def predict_from_image(image):
    # Chuyển ảnh sang grayscale nếu cần thiết
    # if image.mode != "RGB":
    #     image = image.convert("RGB")

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

        train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
        # Lấy tên của nhãn dự đoán từ tập dữ liệu
        predicted_label = train_dataset[predicted_class_idx]

        return predicted_label
    else:
        return "No face detected."


def predict_from_list(images):
    predicted_labels = []
    for image in images:


        # Chuyển ảnh sang numpy array
        image_np = np.array(image)

        # Chuyển ảnh sang grayscale để sử dụng mô hình nhận diện khuôn mặt
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        try:
            gray = Image.fromarray(gray)
            input_image = transform(gray).unsqueeze(0)  # Thêm chiều batch (batch size = 1)

            # Thực hiện dự đoán
            with torch.no_grad():
                output = model(input_image)

            # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
            predicted_class_idx = torch.argmax(output).item()

            train_dataset = {0: 'Khuôn mặt trái tim', 1: 'Khuôn mặt hình chữ nhật/Khuôn mặt dài', 2: 'Khuôn mặt trái xoan', 3: 'Khuôn mặt tròn', 4: 'Khuôn mặt vuông'}
            # Lấy tên của nhãn dự đoán từ tập dữ liệu
            predicted_label = train_dataset[predicted_class_idx]

            predicted_labels.append(predicted_label)
        except:
             predicted_labels.append("No face detected")

    # Đếm số lần xuất hiện của mỗi nhãn dự đoán
    label_counts = Counter(predicted_labels)

    # Lấy nhãn có số lần xuất hiện nhiều nhất
    most_common_label = label_counts.most_common(1)[0][0]

    return most_common_label
def main():
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
                    "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting. "
                    "Also try adjusting detection parameters")
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

                # Download the image
                # face_img = Image.fromarray(face_img)
                # buffered = BytesIO()
                # img.save(buffered, format="JPEG")
                # Creating columns to center button
                col1, col2, col3 = st.columns(3)
                with col1:
                    pass
                with col3:
                    pass
                with col2:
                    if st.button("Predict"):
                       
                        predicted_label = predict_from_face_image(face_img )
                        st.subheader("Hình Dạng Khuôn mặt:")
                        st.markdown(
                            f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                            unsafe_allow_html=True)
        
                        st.markdown('**Ngành Nghề Phù Hợp:**')
                        for career in class_info[predicted_label]['careers']:
                            st.markdown(f"- {career}")
                        st.markdown('**Đặc điểm tính cách:**')
                        st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")
                        

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # ctx = webrtc_streamer(
    #     key="snapshot",
    #     mode=WebRtcMode.SENDRECV,
    #     async_processing=True,
    #     rtc_configuration={
            
    #         "iceServers": [{"urls": ["stun:stun.flashdance.cx:3478"]}],
    #         # "iceServers": token.ice_servers
    #         "iceTransportPolicy": "relay"
    #     },
    #     media_stream_constraints={"video": True, "audio": False},
    #     video_processor_factory=VideoTransformer
    # )
    # if ctx.video_transformer:
    #     if st.button("Predict"):
    #         with ctx.video_transformer.frame_lock:

    #             img_list = ctx.video_transformer.img_list

    #         if img_list is not []:  # put in column form 5 images in a row
    #             predicted_label = predict_from_list(img_list)
    #             st.subheader("Hình Dạng Khuôn mặt:")
    #             st.markdown(
    #                 f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
    #                 unsafe_allow_html=True)

    #             st.markdown('**Ngành Nghề Phù Hợp:**')
    #             for career in class_info[predicted_label]['careers']:
    #                 st.markdown(f"- {career}")
    #             st.markdown('**Đặc điểm tính cách:**')
    #             st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")
    #         else:
    #             st.warning("No faces available yet. Press predict again")
if __name__ == "__main__":
    main()
