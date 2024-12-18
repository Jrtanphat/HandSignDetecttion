from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Tải dữ liệu từ file pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Chuyển dữ liệu sang mảng numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Kiểm tra số lượng mẫu trong dữ liệu
print(f"Total samples: {len(data)}")

# Nếu có đủ dữ liệu, tách thành tập huấn luyện và kiểm tra
if len(data) > 1:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, shuffle=True, stratify=labels)

    # Huấn luyện mô hình
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Dự đoán và đánh giá mô hình
    y_predict = model.predict(x_test)

    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy on test set: {accuracy:.4f}")

    # Lưu mô hình vào file
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
else:
    print("Not enough data to split into train and test sets.")
