import cv2
import numpy as np
import matplotlib.pyplot as plt
import time  # Import the time module

# Load and display the original image
img = cv2.imread("/Users/ullekhpatel/Desktop/Blurry-low-quality-female-portrait-picture.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')
plt.show()
print("Original image shape:", img.shape)

def hierarchical_filling(img, factor):
    """Perform hierarchical filling upscaling."""
    upscaled_img = cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
    low_res = cv2.resize(upscaled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    residual = cv2.subtract(img, low_res)
    upscaled_img += cv2.resize(residual, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
    return upscaled_img

def fast_adaptive_upscaling(img, factor):
    """Perform fast adaptive upscaling."""
    img = img.astype(np.float32)
    img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
    img_residual = cv2.subtract(img, img_blurred)
    upscaled_img = cv2.resize(img_blurred, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
    upscaled_img += cv2.resize(img_residual, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_CUBIC)
    upscaled_img = np.clip(upscaled_img, 0, 255).astype(np.uint8)
    return upscaled_img

def bicubic_interpolation(img, scaling_factor):
    """Perform bicubic interpolation."""
    h, w, c = img.shape
    new_h, new_w = int(h * scaling_factor), int(w * scaling_factor)
    upscaled_img = np.zeros((new_h, new_w, c), dtype=np.uint8)
    a = -0.5

    for i in range(new_h):
        for j in range(new_w):
            x, y = i / scaling_factor, j / scaling_factor
            x1, y1 = int(np.floor(x)), int(np.floor(y))
            u, v = x - x1, y - y1

            p = []
            for k in range(4):
                for l in range(4):
                    px, py = max(0, min(x1 + k - 1, h - 1)), max(0, min(y1 + l - 1, w - 1))
                    p.append(img[px, py])

            A = np.array([[1, u, u**2, u**3],
                          [0, 1, 2*u, 3*u**2],
                          [1, v, v**2, v**3],
                          [0, 1, 2*v, 3*v**2]])

            B = np.zeros((4, c))
            for k in range(c):
                B[:, k] = np.array([p[k][0], p[k][1], p[k][2], 0])

            C = np.array([[a, 0, a, 0],
                          [0, 1, 0, 0],
                          [a, 0, 1-a, 0],
                          [0, 0, 0, a]])

            D = np.dot(np.dot(A, C), B)

            for k in range(c):
                upscaled_img[i, j, k] = int(np.clip(D[:, k].sum(), 0, 255))

    return upscaled_img

def upscale_bilinear(img, scale):
    """Perform bilinear interpolation."""
    width, height = img.shape[1], img.shape[0]
    new_width, new_height = int(width * scale), int(height * scale)
    output = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            x_i, y_i = x / scale, y / scale
            x_0, y_0 = int(x_i), int(y_i)
            x_1, y_1 = min(x_0 + 1, width - 1), min(y_0 + 1, height - 1)
            x_diff, y_diff = x_i - x_0, y_i - y_0

            for c in range(3):
                output[y, x, c] = (1 - x_diff) * (1 - y_diff) * img[y_0, x_0, c] + \
                                  x_diff * (1 - y_diff) * img[y_0, x_1, c] + \
                                  (1 - x_diff) * y_diff * img[y_1, x_0, c] + \
                                  x_diff * y_diff * img[y_1, x_1, c]

    return output

def upscale_nearest(img, scale):
    """Perform nearest neighbor interpolation."""
    width, height = img.shape[1], img.shape[0]
    new_width, new_height = int(width * scale), int(height * scale)
    output = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            x_i, y_i = int(x / scale), int(y / scale)
            output[y, x] = img[y_i, x_i]

    return output

# Perform and display upscaling using different methods
methods = [
    ("Hierarchical Filling", lambda: fast_adaptive_upscaling(img, factor=2)),
    ("Bicubic Interpolation", lambda: bicubic_interpolation(img, scaling_factor=2)),
    ("Bilinear Interpolation", lambda: upscale_bilinear(img, 2)),
    ("Nearest Neighbor", lambda: upscale_nearest(img, 2))
]

plt.figure(figsize=(16, 12))  # Set the figure size for better visualization

times = []  # List to store execution times for each method

for i, (method_name, method_func) in enumerate(methods):
    start_time = time.time()  # Start the timer
    upscaled_img = method_func()
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    times.append(execution_time)  # Append execution time to the list
    
    plt.subplot(2, 2, i + 1)  # Arrange plots in 2x2 grid
    plt.imshow(upscaled_img)
    plt.title(f"{method_name} - Shape: {upscaled_img.shape}\nTime: {execution_time:.4f} seconds")
    plt.axis('off')
    print(f"{method_name} upscaled image shape:", upscaled_img.shape)
    print(f"Time taken: {execution_time:.4f} seconds")

plt.tight_layout()
plt.show()

# Display a comparison of times
for method_name, exec_time in zip([m[0] for m in methods], times):
    print(f"{method_name} execution time: {exec_time:.4f} seconds")