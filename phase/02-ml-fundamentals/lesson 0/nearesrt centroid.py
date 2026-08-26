

"""
================================================================================
BÀI TẬP LỚN: PHÂN LOẠI HOA IRIS BẰNG THUẬT TOÁN NEAREST CENTROID (FROM SCRATCH)
================================================================================

1. BỐI CẢNH VÀ MỤC TIÊU:
   - Dữ liệu: Bộ dữ liệu phân loại 3 loài hoa Iris (0: Setosa, 1: Versicolor, 2: Virginica).
   - Mỗi mẫu gồm 4 đặc trưng (đơn vị: cm):
       + X[:, 0]: Chiều dài đài hoa (Sepal Length)
       + X[:, 1]: Chiều rộng đài hoa (Sepal Width)
       + X[:, 2]: Chiều dài cánh hoa (Petal Length)
       + X[:, 3]: Chiều rộng cánh hoa (Petal Width)
   - Mục tiêu: Xây dựng và đánh giá mô hình Nearest Centroid bằng thư viện NumPy.

2. NGUYÊN LÝ THUẬT TOÁN:
   - Huấn luyện (fit): Tính vector trung bình (centroid) của 4 đặc trưng cho từng lớp:
       C_k = mean(X[y == k], axis=0)
   - Dự đoán (predict): Với mỗi mẫu x, tính khoảng cách Euclid đến tất cả các centroid:
       d(x, C_k) = sqrt(sum((x - C_k)^2))
       Gán nhãn: y_pred = argmin_k (d(x, C_k))

3. YÊU CẦU:
   - Huấn luyện mô hình với tập dữ liệu X_train, y_train được cung cấp sẵn.
   - Dự đoán trên tập dữ liệu X_test.
   - Đánh giá mô hình: In ra nhãn dự đoán và tỷ lệ chính xác (Accuracy).
================================================================================
"""



# code giải 


import numpy as np

#  phần 1 : DỮ LIỆU ĐẦU VÀO (INPUT DATA)
# ==========================================

# Tập huấn luyện (18 mẫu đại diện cho 3 lớp)
X_train = np.array([
    # Lớp 0: Setosa
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    
    # Lớp 1: Versicolor
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3],
    
    # Lớp 2: Virginica
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1]
])

y_train = np.array([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2
])

# Tập kiểm thử (6 mẫu mới cần dự đoán)
X_test = np.array([
    [5.0, 3.4, 1.5, 0.2],  # Thực tế: 0 (Setosa)
    [4.8, 3.4, 1.6, 0.2],  # Thực tế: 0 (Setosa)
    [5.6, 2.9, 3.6, 1.3],  # Thực tế: 1 (Versicolor)
    [6.7, 3.1, 4.4, 1.4],  # Thực tế: 1 (Versicolor)
    [6.4, 2.7, 5.3, 1.9],  # Thực tế: 2 (Virginica)
    [7.2, 3.6, 6.1, 2.5]   # Thực tế: 2 (Virginica)
])

y_test = np.array([0, 0, 1, 1, 2, 2])


# phần 2 : CODE XỬ LÝ 

class Nearest_centroid:
    def __init__ (self):
        self.average_list = []
    def fit(self , X , Y) :
        self.average_list = np.zeros((len(np.unique(Y)) , len(X[0]) ))
        for x , y in zip(X, Y):
            self.average_list[y]+=x
        for i in np.unique(Y):
            self.average_list[i]/=np.sum(Y==i)
            print(self.average_list[i], "\n")
    def predict(self , X):
        min_val = float('inf')
        min_id = 0
        for y in range(len(self.average_list)):
            tong = 0 
            for id , val in enumerate(X):
                tong += (self.average_list[y][id] - val)**2
            min_val , min_id = min((min_val , min_id) , (tong , y))           
        return min_id


        


kiem_tra = Nearest_centroid()
print(2)
kiem_tra.fit(X_train , y_train)
for id , x in enumerate(X_test) :
    if kiem_tra.predict(x) == y_test[id] :
        print ("test ", id , " : CORRECT\n")
    else :
        print("test ", id , " : Wrong\n")
