import cv2
import mediapipe as  mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

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

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý hình ảnh với MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Ảnh để vẽ
                hand_landmarks,  # Đầu ra từ mô hình
                HAND_CONNECTIONS,  # Kết nối đã sửa
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
                x_.append(landmark.x)
                y_.append(landmark.y)

        x1 = int(min(x_) * W) -12
        y1 = int(min(y_) * H) -12

        x2 = int(max(x_) * W) -5
        y2 = int(max(y_) * H) -5

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 -15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)


    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()