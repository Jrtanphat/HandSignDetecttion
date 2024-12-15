import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Kết nối các điểm trên bàn tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Tự định nghĩa HAND_CONNECTIONS nếu cần
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),  # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),  # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Ngón út
]

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# Đường dẫn dữ liệu
DATA_DIR = './data'

# Danh sách chứa dữ liệu và nhãn
data = []
labels = []

# Đọc dữ liệu từ thư mục
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Mảng lưu tọa độ của một hình ảnh
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Xử lý hình ảnh với MediaPipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

            # Kiểm tra nếu số lượng tọa độ không đúng, bỏ qua mẫu này
            if len(data_aux) == 42:  # 21 điểm với (x, y) mỗi điểm
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Skipped an image in {dir_}/{img_path} due to incorrect landmark count.")

# Lưu dữ liệu vào tệp pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data saved successfully!")
