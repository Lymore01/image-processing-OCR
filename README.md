# Text Extraction from Images

## Overview

This project uses Python and various libraries to extract text from images using Optical Character Recognition (OCR). It employs the `easyocr` library to read text and utilizes OpenCV for image processing. The program processes an image to detect text areas and extracts the text, displaying the results.

## Libraries Used

- **easyocr**: A Python library for Optical Character Recognition (OCR).
- **OpenCV (cv2)**: A library for computer vision and image processing.
- **imutils**: A collection of convenience functions for OpenCV.
- **matplotlib**: A plotting library to visualize images.

## Installation

Make sure you have Python installed on your system. You can install the required libraries using `pip`. Run the following command in your terminal:

```bash
pip install easyocr opencv-python imutils matplotlib
```
## Usage
Save your images in the images directory.
Update the image_path variable in the script to point to your image file.
Run the script:
```bash
python your_script_name.py
```
Replace your_script_name.py with the name of your Python script.

## Example Output
After running the script, you will see the output in the console, which will display the extracted text. The processed image with detected rectangles around the text areas will also be shown.

## Sample Output:
```bash
Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.
[([[7, 0], [381, 0], [381, 81], [7, 81]], 'IT20 BOM', 0.7710774043955325)]
```

