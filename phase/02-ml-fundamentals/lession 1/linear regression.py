import math as mt
import numpy as np
import random 

# =====================================================================
# PHẦN 1: SINH DỮ LIỆU GIẢ LẬP (Sao chép hoàn toàn để lấy môi trường test)
# =====================================================================

"""
1. BỐI CẢNH BÀI TOÁN:
Bạn đang làm việc tại bộ phận HR Data Analytics của một Tập đoàn Công nghệ lớn.
Quy trình thỏa thuận lương (deal lương) cho ứng viên hiện tại đang bị cảm tính. 
Để giải quyết việc này, công ty đã thu thập dữ liệu lương thực tế của 100 kỹ sư AI
đang làm việc tại tập đoàn, bao gồm hai thông tin chính:
  - Số năm kinh nghiệm của kỹ sư (từ 0 đến 10 năm) -> Ký hiệu là X_data.
  - Mức lương tháng thực tế (đơn vị: Triệu VNĐ)   -> Ký hiệu là y_data.

2. MỤC TIÊU:
Xây dựng mô hình Hồi quy tuyến tính từ con số 0 để tự động hóa việc đề xuất mức 
lương hợp lý cho các ứng viên mới dựa trên số năm kinh nghiệm của họ.

3. Ý NGHĨA CÁC THAM SỐ CẦN TÌM:
  - Hệ số chặn (b): Mức lương khởi điểm tối thiểu của sinh viên mới ra trường (0 năm kinh nghiệm).
  - Trọng số (w): Mức lương cộng thêm trung bình cho mỗi 1 năm kinh nghiệm tích lũy.
  - Sai số (y_hat - y): Sự chênh lệch giữa lương đề xuất (y_hat) và lương thực tế (y)
    do khả năng đàm phán cá nhân hoặc các chứng chỉ bổ sung của từng nhân sự.

4. CÁC NHIỆM VỤ BẠN CẦN HOÀN THÀNH:
  - TODO 1 (predict): Nhận vào X, tính lương đề xuất: y_hat = w * x + b
  - TODO 2 (compute_cost): Tính sai số bình phương trung bình (MSE) của bảng lương đề xuất.
  - TODO 3 (compute_gradients): Tính đạo hàm riêng dw và db để tìm hướng điều chỉnh w và b.
  - TODO 4 (fit): Chạy vòng lặp để cập nhật w và b sau mỗi lần "dự đoán lại" dữ liệu.
  - TODO 5 (r_squared): Tính chỉ số R-squared để đo lường độ chính xác của mô hình.
  - TODO 6 (solve_normal_equation): Dùng công thức ma trận của CFO để giải trực tiếp ra kết quả.

================================================================================
"""
# Cố định hạt giống ngẫu nhiên để dữ liệu không đổi giữa các lần chạy
random.seed(42)

# Thiết lập đáp án ẩn (Ground Truth)
TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 1000

# Sinh 100 điểm dữ liệu (X: diện tích, y: giá nhà có cộng nhiễu Gauss)
X_data = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
Y_data = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X_data]

# =====================================================================
# PHẦN 2: TỰ VIẾT LẠI LỚP HỒI QUY TUYẾN TÍNH 
# =====================================================================

class linear_regression :
    def __init__(self , learning_rate = 0.01 ) :
        self.w = 0.0
        self.b = 0.
        self.lr = learning_rate

    def predict_y(self , x):
        return self.w*x+self.b
    def gradient_desent(self,X_data , Y_data):
        dMSE_dw = 0.0
        dMSE_db = 0.0
        for id , val_x in enumerate(X_data):
            y_hat = self.predict_y(val_x)
            dMSE_dw += (y_hat - Y_data[id])*val_x
            dMSE_db += (y_hat - Y_data[id])
        dMSE_dw *= (2/N_SAMPLES)
        dMSE_db *= (2/N_SAMPLES)
        return dMSE_dw , dMSE_db
    def update(self,X_data , Y_data):
        dMSE_dw,dMSE_db = self.gradient_desent(X_data , Y_data)
        self.w -= self.lr*dMSE_dw
        self.b -= self.lr*dMSE_db
    def r_squared(self ,X_data , Y_data):
        y_bar = sum(Y_data)/len(Y_data)
        tuso = 0.0
        mauso= 0.0
        for id , val_x in enumerate(X_data):
            tuso += (self.predict_y(val_x) - Y_data[id])**2
            mauso += (y_bar - Y_data[id])**2
        R_2 = 1.0 - tuso/mauso
        return R_2

    def fit(self , X_data , Y_data , epochs = 0):
        for i in range (epochs):
            self.update(X_data , Y_data)


# =====================================================================
# PHẦN 3: CHẠY THỬ VÀ ĐỐI CHIẾU KẾT QUẢ (Sao chép để kiểm tra code tự viết)
# =====================================================================

if __name__ == "__main__":
    print("--- 1. HUẤN LUYỆN BẰNG GRADIENT DESCENT ---")
    model = linear_regression(learning_rate=0.005)
    model.fit(X_data, Y_data, epochs=1000)
    print(f"Kết quả Gradient Descent: w = {model.w:.4f}, b = {model.b:.4f}")
    print(f"Độ chính xác R2 Score:   {model.r_squared(X_data, Y_data):.4f}")
 
    print(f"\n--- ĐÁP ÁN ẨN THỰC TẾ ---:   w = {TRUE_W}, b = {TRUE_B}")


