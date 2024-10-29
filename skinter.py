import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def is_homogeneous(region, threshold):
    """Kiểm tra xem một vùng có đồng nhất không dựa trên độ sáng trung bình."""
    mean = np.mean(region)
    return np.all(np.abs(region - mean) < threshold)

def kmeans_merge(image, k=4):
    """Hợp nhất các vùng sử dụng thuật toán K-means."""
    Z = image.reshape((-1, 1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    result_image = centers[labels.flatten()]
    result_image = result_image.reshape(image.shape)

    return result_image

def split_merge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 19

    def recursive_split(y_start, y_end, x_start, x_end):
        region = gray[y_start:y_end, x_start:x_end]

        if (y_end - y_start <= 1) or (x_end - x_start <= 1) or is_homogeneous(region, threshold):
            mean_value = np.mean(region)
            return np.full(region.shape, mean_value, dtype=np.uint8)

        mid_y = (y_start + y_end) // 2
        mid_x = (x_start + x_end) // 2

        top_left = recursive_split(y_start, mid_y, x_start, mid_x)
        top_right = recursive_split(y_start, mid_y, mid_x, x_end)
        bottom_left = recursive_split(mid_y, y_end, x_start, mid_x)
        bottom_right = recursive_split(mid_y, y_end, mid_x, x_end)

        return merge_images(top_left, top_right, bottom_left, bottom_right)

    def merge_images(top_left, top_right, bottom_left, bottom_right):
        top = np.hstack((top_left, top_right))
        bottom = np.hstack((bottom_left, bottom_right))
        combined = np.vstack((top, bottom))
        return kmeans_merge(combined)

    processed_image = recursive_split(0, gray.shape[0], 0, gray.shape[1])

    if processed_image.shape != gray.shape:
        processed_image = cv2.resize(processed_image, (gray.shape[1], gray.shape[0]))

    return gray, processed_image

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        original_image, result = split_merge(img)

        # Hiển thị ảnh gốc và ảnh đã xử lý
        show_images(original_image, result)

def show_images(original_image, result):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    original_image = Image.fromarray(original_image)
    result_image = Image.fromarray(result)

    original_image = ImageTk.PhotoImage(original_image)
    result_image = ImageTk.PhotoImage(result_image)

    panel_original.config(image=original_image)
    panel_original.image = original_image

    panel_result.config(image=result_image)
    panel_result.image = result_image

# Tạo giao diện tkinter
root = tk.Tk()
root.title("Image Segmentation with K-means")

# Nút tải hình ảnh
btn_load = tk.Button(root, text="Load Image", command=load_image)
btn_load.pack()

# Panels để hiển thị ảnh
panel_original = tk.Label(root)
panel_original.pack(side="left", padx=10, pady=10)

panel_result = tk.Label(root)
panel_result.pack(side="right", padx=10, pady=10)

root.mainloop()
