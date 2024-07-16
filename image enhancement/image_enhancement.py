import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)



URL = r"https://www.dropbox.com/s/0oe92zziik5mwhf/opencv_bootcamp_assets_NB4.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB4.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)



img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Display 18x18 pixel image.
plt.imshow(img_rgb)
plt.show()

# ADDITION or BRIGHTNESS

matrix = np.ones(img_rgb.shape, dtype="uint8") * 50
img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)
# Show the images
plt.imshow(img_rgb_darker);  plt.title("Darker")  ; plt.show();
plt.imshow(img_rgb);         plt.title("Original"); plt.show();
plt.imshow(img_rgb_brighter);plt.title("Brighter"); plt.show();

# MULTIPLICATION or CONTRAST
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))

# Show the images
plt.imshow(img_rgb_darker);  plt.title("Lower Contrast"); plt.show()
plt.imshow(img_rgb);         plt.title("Original"); plt.show()
plt.imshow(img_rgb_brighter);plt.title("Higher Contrast"); plt.show()

# the distorted colors on the higher contrast picture are due to values > 255 being created by the multiplication

img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0 ,255))
plt.imshow(img_rgb_higher);plt.title("Higher Contrast With Clip"); plt.show()

# IMAGE THRESHOLDING
# retval, dst = cv2.threshold( src, thresh, maxval, type[, dst])

img_read = cv2.imread("building-windows.jpg", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)
# Show the images
plt.imshow(img_read, cmap="gray");  plt.title("Original"); plt.show()
plt.imshow(img_thresh, cmap="gray");plt.title("Thresholded"); plt.show()

# Application: Sheet music reader
# suppose you wanted an app that could read (decode) sheet music. Similar to Optical Character Recognition (OCR) for text documents. One of the first steps to this is to reduce unnecessary image data by thresholding

img_read = cv2.imread("Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)
# Perform global thresholding
retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)

# Perform global thresholding
retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

# Show the images

plt.imshow(img_read,        cmap="gray");  plt.title("Original"); plt.show()
plt.imshow(img_thresh_gbl_1,cmap="gray");  plt.title("Thresholded (global: 50)"); plt.show()
plt.imshow(img_thresh_gbl_2,cmap="gray");  plt.title("Thresholded (global: 130)"); plt.show()
plt.imshow(img_thresh_adp,  cmap="gray");  plt.title("Thresholded (adaptive)"); plt.show()


# BITWISE OPERATIONS ON TWO IMAGES

img_rec = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("circle.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img_rec, cmap="gray");plt.title("Rectangle"); plt.show()
plt.imshow(img_cir, cmap="gray");plt.title("Circle"); plt.show()

result = cv2.bitwise_and(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray");plt.title("Bitwise AND"); plt.show()

result = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray");plt.title("Bitwise OR"); plt.show()

result = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray");plt.title("Bitwise XOR"); plt.show()

# READ FOREGROUND IMAGE
img_bgr = cv2.imread("coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.title("Coca Cola"); plt.show()
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]


# Read in image of color cheackerboad background
img_background_bgr = cv2.imread("checkerboard_color.png")
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

# Set desired width (logo_w) and maintain image aspect ratio
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))

# Resize background image to sae size as logo image
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

plt.imshow(img_background_rgb); plt.title("Background"); plt.show()

# gray scale coca cola logo
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_mask, cmap="gray"); plt.title("Gray Coke logo"); plt.show()

# invert mask to flip colors of FG and BG
img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray"); plt.title("Inverted Gray Coke Logo"); plt.show()

# apply background on mask
img_background = cv2.bitwise_and( img_background_rgb, img_background_rgb, mask=img_mask)
plt.imshow(img_background); plt.title("With Background Mask"); plt.show()

# Isolate foreground (red from original image) using the inverse mask
img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.imshow(img_foreground); plt.title("Original background with inverse foreground"); plt.show()



# Add the two previous results obtain the final result
result = cv2.add(img_background, img_foreground)
plt.imshow(result); plt.title("Previous two results added"); plt.show()
cv2.imwrite("logo_final.png", result[:, :, ::-1])
