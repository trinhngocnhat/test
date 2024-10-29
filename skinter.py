import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def is_homogeneous(region, threshold):
    """Kiểm tra xem một vùng có đồng nhất không dựa trên độ sáng trung bình."""
    mean = np.mean(region, axis=(0, 1))  # Tính trung bình màu
    return np.all(np.abs(region - mean) < threshold)


def merge_regions(region):
    """Hợp nhất vùng bằng cách tính trung bình màu."""
    mean_color = np.mean(region, axis=(0, 1)).astype(np.uint8)
    return np.full(region.shape, mean_color, dtype=np.uint8)


def split_merge(img):
    # Thiết lập ngưỡng để phân chia các vùng
    threshold = 30

    # Hàm chia và hợp
    def recursive_split(y_start, y_end, x_start, x_end):
        region = img[y_start:y_end, x_start:x_end]

        # Nếu vùng nhỏ hơn hoặc bằng 1 pixel hoặc đồng nhất, trả về vùng đó
        if (y_end - y_start <= 1) or (x_end - x_start <= 1) or is_homogeneous(region, threshold):
            return merge_regions(region)

        # Chia vùng thành 4 phần
        mid_y = (y_start + y_end) // 2
        mid_x = (x_start + x_end) // 2

        # Đệ quy cho 4 vùng
        top_left = recursive_split(y_start, mid_y, x_start, mid_x)
        top_right = recursive_split(y_start, mid_y, mid_x, x_end)
        bottom_left = recursive_split(mid_y, y_end, x_start, mid_x)
        bottom_right = recursive_split(mid_y, y_end, mid_x, x_end)

        # Hợp các vùng lại
        return combine_images(top_left, top_right, bottom_left, bottom_right)

    def combine_images(top_left, top_right, bottom_left, bottom_right):
        """Hợp các vùng lại thành một hình ảnh duy nhất."""
        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        combined = np.vstack((top, bottom))
        return combined

    # Thực hiện chia và hợp
    processed_image = recursive_split(0, img.shape[0], 0, img.shape[1])

    return img, processed_image  # Trả về ảnh gốc và ảnh đã xử lý


def load_image():
    """Tải hình ảnh từ tệp tin và xử lý."""
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        original_image, result = split_merge(img)

        # Chuyển đổi ảnh từ BGR sang RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Hiển thị ảnh gốc
        display_image(original_image, original_label)

        # Hiển thị ảnh đã xử lý
        display_image(result, result_label)


def display_image(image, label):
    """Hiển thị ảnh trong label."""
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    label.config(image=image)
    label.image = image  # Giữ tham chiếu để không bị garbage collection


# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Ứng dụng Phân chia và Hợp vùng Ảnh")

# Nút tải ảnh
load_button = tk.Button(root, text="Tải hình ảnh", command=load_image)
load_button.pack()

# Label để hiển thị ảnh gốc
original_label = tk.Label(root)
original_label.pack(side=tk.LEFT)

# Label để hiển thị ảnh đã xử lý
result_label = tk.Label(root)
result_label.pack(side=tk.RIGHT)

# Chạy ứng dụng
root.mainloop()
