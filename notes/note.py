# this file contains some notes and concepts on image processing, collected from different sources while learning 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

def process_image(file_path):
    # Load the image in BGR format
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization - cv2 displays image in BGR

    # Convert to grayscale - enhances features for processing, helps in edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize - helps to reduce computational load, normalization (used in creating thumbnails)
    resized_image = cv2.resize(image, (100, 100))  # (width, height)

    # Crop - extract region of interest (ROI) from an image
    cropped_image = image[50:200, 100:300]  # y:y+h, x:x+w

    # Blur - reduces noise and detail
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)  # (kernel_size, sigma)

    # Edge detection - detects edges
    edges = cv2.Canny(gray_image, 100, 200)  # (low_threshold, high_threshold)

    # Thresholding - binarizes the image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)  # (threshold_value, max_value, type)

    # Morphological operation (dilation) - preprocess image based on shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Define the kernel
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)  # (image, kernel, iterations)

    # Contours detection - find contours of objects in a binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # (image, contours, contourIdx, color, thickness)

    # Image transformation (rotation)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # (center, angle, scale)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))  # (image, transformation_matrix, output_size)

    # Feature detection and description - detects keypoints and computes descriptors
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)  # Detect keypoints and descriptors
    feature_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))  # Draw keypoints

    # Histogram equalization - enhances contrast in grayscale images
    equalized_image = cv2.equalizeHist(gray_image)  # Apply histogram equalization
    
    # Display results
    # subplots(row, col, size for each plot)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    # axes[0, 0 or row1, col1]
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[1, 0].imshow(resized_image)
    axes[1, 0].set_title('Resized')
    axes[1, 1].imshow(blurred_image)
    axes[1, 1].set_title('Blurred')
    axes[2, 0].imshow(edges, cmap='gray')
    axes[2, 0].set_title('Edges')
    axes[2, 1].imshow(dilated_image, cmap='gray')
    axes[2, 1].set_title('Dilated')
    
    # .flat allows use to access and manipulate each subplot
    for ax in axes.flat:
        ax.axis('off')
    plt.show()

# Run the function
# process_image('images\car.jpg')



# ? rectangle detection in an image

def count_rectangles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image
    """ 
    60 - threshold value - values below this are set to 0 black, values above they are set to white
    255 - maximum value 
    """
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    edged = cv2.Canny(binary, 30, 200)
    # Find contours
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    rectangle_count = 0
    cont = []
    for contour in contours:
        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour has 4 points
        if len(approx) == 2:
            cont.append(contour)
            # Further check if the shape is a rectangle by checking the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                rectangle_count += 1
                # Draw the contour in the image for visualization
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
    
    # Display the image with rectangles highlighted
    plt.imshow(binary, cmap='gray')
    plt.title('Rectangles Detected')
    # plt.axis('off')
    plt.show()
    
    return rectangle_count

# Test the function
# rect_count = count_rectangles('images/rect.jpg')
# print(f"Number of rectangles: {rect_count}")
# print(count_rectangles('images/rect.jpg'))

# ? text recognition
import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext("images/three.png")
print(result)

# ? Creating mask
""" import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('path_to_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define a contour (location)
location = np.array([[[300, 540]], [[306, 589]], [[543, 592]], [[538, 543]]], dtype=np.int32)

# Create a mask of zeros (black image)
mask = np.zeros(gray.shape, np.uint8)

# Draw the contour on the mask
cv2.drawContours(mask, [location], 0, 255, -1)

# Apply the mask to the original image
new_image = cv2.bitwise_and(img, img, mask=mask)

# Display the result
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Masked Image')
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show() """


# https://www.youtube.com/watch?v=Yf3bQJfh9yg