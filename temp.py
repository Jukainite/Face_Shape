import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
import time
# Load lại mô hình đã được huấn luyện
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

        train_dataset = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
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

            train_dataset = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
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



# Định nghĩa transformer cho webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.frame_list = []
        self.play_state = True
    def transform(self, frame):
        if len(self.frame_list) >= 25:  #
           self.consumer.stop()


        img = frame.to_ndarray(format="bgr24")
        self.time= time.time()
        # Tạo một bộ phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # # Nhận diện khuôn mặt trong frame
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        # Vẽ hình chữ nhật quanh mỗi khuôn mặt và hiển thị kết quả
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y:y + h, x:x + w]
            self.frame_list.append(face)
        if len(self.frame_list) >=10:
            st.stop()

        return img




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
    }
}
# Định nghĩa giao diện Streamlit
def main():
    st.title("Face Shape Prediction")

    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Start'):
            predicted_label = predict_from_image(image)

            st.subheader("Hình Dạng Khuôn mặt:")
            st.markdown(
                f"<p style='text-align:center; font-size:60px; color:blue'><strong>{predicted_label}</strong></p>",
                unsafe_allow_html=True)

            st.markdown('**Ngành Nghề Phù Hợp:**')
            for career in class_info[predicted_label]['careers']:
                st.markdown(f"- {career}")
            st.markdown('**Đặc điểm tính cách:**')
            st.write("Để xem lí giải cụ thể, bạn hãy đăng kí gói vip của thần số học ! ♥ ♥ ♥")

if __name__ == "__main__":
    main()
