import cv2
import numpy as np


def is_homogeneous(region, threshold):
    """Kiểm tra xem một vùng có đồng nhất không dựa trên độ sáng trung bình."""
    mean = np.mean(region)
    return np.all(np.abs(region - mean) < threshold)


def kmeans_merge(image, k=4):
    """Hợp nhất các vùng sử dụng thuật toán K-means."""
    # Chuyển đổi hình ảnh thành một mảng 2D
    Z = image.reshape((-1, 1))
    Z = np.float32(Z)

    # Thiết lập các tiêu chí và thực hiện K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Tạo hình ảnh hợp nhất từ các nhãn
    centers = np.uint8(centers)
    result_image = centers[labels.flatten()]
    result_image = result_image.reshape(image.shape)

    return result_image


def split_merge(img):
    # Chuyển đổi hình ảnh thành ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thiết lập ngưỡng để phân chia các vùng
    threshold = 19

    # Hàm chia và hợp
    def recursive_split(y_start, y_end, x_start, x_end):
        region = gray[y_start:y_end, x_start:x_end]

        # Nếu vùng nhỏ hơn hoặc bằng 1 pixel hoặc đồng nhất, trả về vùng đó
        if (y_end - y_start <= 1) or (x_end - x_start <= 1) or is_homogeneous(region, threshold):
            mean_value = np.mean(region)
            return np.full(region.shape, mean_value, dtype=np.uint8)

        # Chia vùng thành 4 phần
        mid_y = (y_start + y_end) // 2
        mid_x = (x_start + x_end) // 2

        # Đệ quy cho 4 vùng
        top_left = recursive_split(y_start, mid_y, x_start, mid_x)
        top_right = recursive_split(y_start, mid_y, mid_x, x_end)
        bottom_left = recursive_split(mid_y, y_end, x_start, mid_x)
        bottom_right = recursive_split(mid_y, y_end, mid_x, x_end)

        # Kết hợp các vùng lại
        return merge_images(top_left, top_right, bottom_left, bottom_right)

    def merge_images(top_left, top_right, bottom_left, bottom_right):
        """Hợp các vùng lại thành một hình ảnh duy nhất."""
        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        combined = np.vstack((top, bottom))

        # Sử dụng K-means để hợp nhất
        return kmeans_merge(combined)

    # Thực hiện chia và hợp
    processed_image = recursive_split(0, gray.shape[0], 0, gray.shape[1])

    # **Thêm bước điều chỉnh kích thước nếu cần**
    if processed_image.shape != gray.shape:
        processed_image = cv2.resize(processed_image, (gray.shape[1], gray.shape[0]))

    return gray, processed_image  # Trả về ảnh gốc và ảnh đã xử lý


# Đọc hình ảnh
img = cv2.imread('D:\\Image\\test6.jpg')

# Thực hiện chia và hợp vùng
original_image, result = split_merge(img)

# Hiển thị kết quả
cv2.imshow('Anh goc', original_image)
cv2.imshow('Sau khi tach va hop', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
