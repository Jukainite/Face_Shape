import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
import time
# Load lại mô hình đã được huấn luyện
model_path = r"face_shape_classifier.pth"
device = torch.device('cpu')  # Sử dụng CPU
model = torchvision.models.efficientnet_b4(pretrained=False)
num_classes = 5
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
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


# Định nghĩa chức năng dự đoán qua webcam
def predict_from_webcam():
    predicted_labels = []

    # Mở webcam
    cap = cv2.VideoCapture(0)

    # Thời gian chạy webcam (10 giây)
    end_time = time.time() + 5

    while time.time() < end_time:
        # Đọc frame từ webcam
        ret, frame = cap.read()

        # Chuyển đổi frame sang ảnh grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt trong frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Vẽ hình chữ nhật xung quanh các khuôn mặt và thực hiện dự đoán
        if len(faces)>0:
            try:
                for (x, y, w, h) in faces:
                    # Vẽ hình chữ nhật xung quanh khuôn mặt
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Chụp ảnh khuôn mặt và chuyển đổi thành tensor
                    face_img = frame[y:y + h, x:x + w]
                    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    input_image = transform(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(input_image)

                    # Lấy chỉ số có giá trị lớn nhất là nhãn dự đoán
                    predicted_class_idx = torch.argmax(output).item()
                    train_dataset = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
                    # Lấy tên của nhãn dự đoán từ tập dữ liệu
                    predicted_label = train_dataset[predicted_class_idx]
                    cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    # Lưu nhãn dự đoán vào danh sách


                    predicted_labels.append(predicted_label)
            except:
                predicted_labels.append("No face detected")
        else:
            predicted_labels.append("No face detected")

        # Hiển thị frame
        cv2.imshow('Face Detection', frame)
        # st.image(frame, channels="BGR", use_column_width=True)

        # Nhấn 'q' để thoát khỏi vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Đóng webcam và cửa sổ
    cap.release()
    cv2.destroyAllWindows()

    # Đếm số lần xuất hiện của mỗi nhãn dự đoán
    label_counts = Counter(predicted_labels)

    # Lấy nhãn có số lần xuất hiện nhiều nhất
    most_common_label = label_counts.most_common(1)[0][0]

    return most_common_label


# Định nghĩa giao diện Streamlit
def main():
    st.title("Face Shape Prediction")

    # Lựa chọn phương pháp dự đoán
    prediction_method = st.radio("Choose prediction method:", ("Webcam", "Image"))

    if prediction_method == "Webcam":
        st.write("Press 'q' to stop the webcam prediction after 10 seconds.")
        most_common_label = predict_from_webcam()
        st.write("Most common label:", most_common_label)
    else:
        image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            image = Image.open(image_file)
            predicted_label = predict_from_image(image)
            st.write("Predicted label:", predicted_label)


if __name__ == "__main__":
    main()
