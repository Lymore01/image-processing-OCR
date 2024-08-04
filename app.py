# Import libraries
import easyocr 
import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np

def read_text_from_image(file_path):
    # Read the image from a file
    image = cv2.imread(file_path)
    
    # Resize image for better handling
    image = imutils.resize(image, width=600)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise
    image_blur = cv2.bilateralFilter(image_gray, 11, 17, 17)
    
    # Apply binary threshold
    _, binary = cv2.threshold(image_blur, 80, 255, cv2.THRESH_BINARY)
    
    # Edge detection
    image_edge = cv2.Canny(binary, 30, 200)
    
    # Find contours
    contours = cv2.findContours(image_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Initialize a variable to store rectangle location
    location = None
    
    for contour in contours:
        # Approximate contour to polygon
        approx = cv2.approxPolyDP(contour, 8, True)
        if len(approx) == 4:
            location = approx
            break
            
    # Draw the detected rectangle on the image
    if location is not None:
        cv2.drawContours(image_rgb, [location], -1, (0, 255, 0), 3)
        mask = np.zeros(image_gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = image_gray[x1:x2+1, y1:y2+1]
    
        # read the text from the cropped image
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)
        
        # Display the image with detected rectangles
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Rectangles')
        plt.axis('off')
        plt.show()

# Path to your image
image_path = "images/cartwo.png"    
read_text_from_image(image_path)
