import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu từ tệp Iris
data_dir = r"C:\Users\duyba\Downloads\Iris.csv"  # Đường dẫn tới file CSV
iris_data = pd.read_csv(data_dir)

# Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = iris_data.drop(columns=['Species'])  # Các đặc trưng
y = iris_data['Species']  # Nhãn (loại hoa)

# Mã hóa nhãn thành số
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Chuyển đổi nhãn thành số

# Chia tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Khởi tạo các bộ phân loại
svm_classifier = SVC(kernel='linear')  # Bộ phân loại SVM
dt_classifier = DecisionTreeClassifier()  # Bộ phân loại Decision Tree
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Bộ phân loại KNN

# Huấn luyện các bộ phân loại
svm_classifier.fit(X_train, y_train)  # Huấn luyện SVM
dt_classifier.fit(X_train, y_train)  # Huấn luyện Decision Tree
knn_classifier.fit(X_train, y_train)  # Huấn luyện KNN

# Dự đoán nhãn cho tập kiểm tra
svm_predictions = svm_classifier.predict(X_test)  # Dự đoán bằng SVM
dt_predictions = dt_classifier.predict(X_test)  # Dự đoán bằng Decision Tree
knn_predictions = knn_classifier.predict(X_test)  # Dự đoán bằng KNN

# Tạo biểu đồ phân tán cho hai đặc trưng
plt.figure(figsize=(15, 10))  # Kích thước của biểu đồ

# Chọn hai đặc trưng để vẽ biểu đồ (ví dụ: SepalLengthCm và SepalWidthCm)
feature1 = 'SepalLengthCm'
feature2 = 'SepalWidthCm'

# Hàm để vẽ biểu đồ dự đoán
def plot_predictions(X, predictions, title):
    plt.scatter(X[feature1], X[feature2], c=predictions, cmap='viridis', edgecolor='k', s=100)
    plt.title(title)  # Tiêu đề của biểu đồ
    plt.xlabel(feature1)  # Nhãn trục x
    plt.ylabel(feature2)  # Nhãn trục y
    plt.grid(True)  # Hiển thị lưới

# Vẽ biểu đồ dự đoán của SVM
plt.subplot(1, 3, 1)  # Tạo lưới 1 hàng 3 cột
plot_predictions(X_test, svm_predictions, 'SVM Predictions')  # Gọi hàm vẽ cho SVM

# Vẽ biểu đồ dự đoán của Decision Tree
plt.subplot(1, 3, 2)  # Tạo lưới 1 hàng 3 cột
plot_predictions(X_test, dt_predictions, 'Decision Tree Predictions')  # Gọi hàm vẽ cho Decision Tree

# Vẽ biểu đồ dự đoán của KNN
plt.subplot(1, 3, 3)  # Tạo lưới 1 hàng 3 cột
plot_predictions(X_test, knn_predictions, 'KNN Predictions')  # Gọi hàm vẽ cho KNN

# Hiển thị các biểu đồ
plt.tight_layout()  # Tự động điều chỉnh khoảng cách giữa các biểu đồ
plt.show()  # Hiển thị biểu đồ