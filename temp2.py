import streamlit as st
import numpy as np
import pandas as pd
import threading
from typing import Union
import cv2
import av
import os
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, ClientSettings, RTCConfiguration
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
import time
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
train_dataset = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

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

# Định nghĩa hàm dự đoán qua ảnh
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

    title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Nhân Tướng Học Khuôn Mặt</p>'
    st.markdown(title, unsafe_allow_html=True)

    st.markdown(
        "Luư Ý Khi sử dụng:"
        " Bạn hãy mở camera và để app xác định khuôn mặt của bạn. Khi phát hiện ra nó sẽ khoanh vùng khuôn mặt. "
        " Khi khung xuất hiện hãy đợi 1 tí (khoảng 3s xuất hiện khung) rồi nhấn nút predict phía dưới")



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


    # ctx = webrtc_streamer(key="snapshot", client_settings=WEBRTC_CLIENT_SETTINGS,video_processor_factory=VideoTransformer)
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    media_stream_constraints = {"video": True, "audio": False}
    
    ctx = webrtc_streamer(
        key="snapshot",
        # rtc_configuration=rtc_configuration,
        media_stream_constraints=media_stream_constraints,
        video_processor_factory=VideoTransformer
    )
    if ctx.video_transformer:
        if st.button("Predict"):
            with ctx.video_transformer.frame_lock:

                img_list = ctx.video_transformer.img_list

            if img_list is not []:  # put in column form 5 images in a row
                predicted_label = predict_from_list(img_list)
                st.subheader("Hình Dạng Khuôn mặt:")
                st.markdown(
                    f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                    unsafe_allow_html=True)

                st.markdown('**Ngành Nghề Phù Hợp:**')
                for career in class_info[predicted_label]['careers']:
                    st.markdown(f"- {career}")
                st.markdown('**Đặc điểm tính cách:**')
                st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")
            else:
                st.warning("No faces available yet. Press predict again")
if __name__ == "__main__":
    main()
