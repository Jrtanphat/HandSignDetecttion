from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Chuyển dữ liệu sang mảng numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Tách dữ liệu
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Dự đoán và đánh giá
y_predict = model.predict(x_test)

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
