# Import thư viện OpenCV để xử lý ảnh và video
import cv2
# Import thư viện numpy để làm việc với mảng đa chiều
import numpy as np

# Hàm callback bắt buộc cho trackbar (không làm gì cả)
def nothing(x):
    pass

# Hàm chính xử lý phát hiện màu
def main():
    # Tạo cửa sổ điều khiển (chứa các thanh trượt trackbar)
    # cv2.WINDOW_NORMAL cho phép thay đổi kích thước cửa sổ
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    # Tạo cửa sổ hiển thị kết quả phát hiện màu
    cv2.namedWindow("Color Detection", cv2.WINDOW_NORMAL)

    # Tạo các thanh trượt (trackbar) để điều chỉnh ngưỡng màu HSV
    # Trackbar cho giá trị Hue (màu sắc) nhỏ nhất (0-179)
    cv2.createTrackbar("lowH", "Controls", 0, 179, nothing)
    # Trackbar cho giá trị Hue lớn nhất
    cv2.createTrackbar("highH", "Controls", 179, 179, nothing)
    
    # Trackbar cho giá trị Saturation (độ bão hòa) nhỏ nhất (0-255)
    cv2.createTrackbar("lowS", "Controls", 0, 255, nothing)
    # Trackbar cho giá trị Saturation lớn nhất
    cv2.createTrackbar("highS", "Controls", 255, 255, nothing)
    
    # Trackbar cho giá trị Value (độ sáng) nhỏ nhất (0-255)
    cv2.createTrackbar("lowV", "Controls", 0, 255, nothing)
    # Trackbar cho giá trị Value lớn nhất
    cv2.createTrackbar("highV", "Controls", 255, 255, nothing)

    # Mở camera mặc định (camera 0)
    cap = cv2.VideoCapture(0)
    
    # Kiểm tra xem camera có mở thành công không
    if not cap.isOpened():
        print("Không thể mở camera!")
        return  # Thoát hàm nếu không mở được camera

    # Vòng lặp chính để xử lý từng khung hình
    while True:
        # Đọc một khung hình từ camera
        # ret: True/False tùy thuộc vào việc đọc khung hình có thành công không
        # frame: khung hình đọc được
        ret, frame = cap.read()
    
        
        # Nếu không đọc được khung hình thì thoát vòng lặp
        if not ret:            break

        # Lấy giá trị hiện tại từ các trackbar
        ilowH = cv2.getTrackbarPos("lowH", "Controls")  # Giá trị Hue nhỏ nhất
        ihighH = cv2.getTrackbarPos("highH", "Controls")  # Giá trị Hue lớn nhất
        ilowS = cv2.getTrackbarPos("lowS", "Controls")  # Giá trị Saturation nhỏ nhất
        ihighS = cv2.getTrackbarPos("highS", "Controls")  # Giá trị Saturation lớn nhất
        ilowV = cv2.getTrackbarPos("lowV", "Controls")  # Giá trị Value nhỏ nhất
        ihighV = cv2.getTrackbarPos("highV", "Controls")  # Giá trị Value lớn nhất

        # Chuyển đổi không gian màu từ BGR (OpenCV mặc định) sang HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Tạo mảng numpy chứa giá trị ngưỡng dưới (H min, S min, V min)
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        # Tạo mảng numpy chứa giá trị ngưỡng trên (H max, S max, V max)
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        
        # Tạo mặt nạ (mask) - các pixel nằm trong khoảng ngưỡng sẽ có giá trị 255, ngoài khoảng là 0
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        
        # Áp dụng mặt nạ lên ảnh gốc để chỉ giữ lại các pixel thỏa mãn điều kiện
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Hiển thị kết quả phát hiện màu
        cv2.imshow('Color Detection', result)
        
        # Nếu nhấn phím 'q' thì thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera
    cap.release()
    # Đóng tất cả cửa sổ OpenCV
    cv2.destroyAllWindows()

# Đoạn code này đảm bảo hàm main() chỉ được chạy khi file được thực thi trực tiếp
# chứ không phải khi được import như một module
if __name__ == "__main__":
    main()