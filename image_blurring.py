import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("cameraman.png")

# Check if image was loaded
if image is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# 1. Gaussian Blurring
# -------------------------------
gaussian = cv2.GaussianBlur(image, (7, 7), 0)

# -------------------------------
# 2. Median Blurring
# -------------------------------
median = cv2.medianBlur(image, 7)

# -------------------------------
# 3. Bilateral Blurring
# -------------------------------
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# Convert results to RGB
gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

# -------------------------------
# Display Results
# -------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Blurring")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(median_rgb)
plt.title("Median Blurring")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(bilateral_rgb)
plt.title("Bilateral Blurring")
plt.axis("off")

plt.tight_layout()
plt.show()