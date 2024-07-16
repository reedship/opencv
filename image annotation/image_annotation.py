import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)

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

URL = r"https://www.dropbox.com/s/48hboi1m4crv1tl/opencv_bootcamp_assets_NB3.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB3.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Read in an image
image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)

# Display the original image
plt.imshow(image[:, :, ::-1])
plt.show()


# to draw a line, use cv2.line()
# cv2.line(src, pt1, pt2, color[, thickness, lineType])

imageLine = image.copy()
cv2.line(imageLine, (200,100), (400,100), (255,0,255),
         thickness=5, lineType=cv2.LINE_AA)
plt.imshow(imageLine[:,:,::-1])
plt.show()


# to draw a circle, use cv2.circle()
# cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])

imageCircle = image.copy()
cv2.circle(imageCircle, (900,500), 100, (0,0,255),
           thickness=5, lineType=cv2.LINE_AA)
plt.imshow(imageCircle[:,:,::-1])
plt.show()


# to draw a rectangle use cv2.rectangle()
# cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (500, 100), (700, 600),
              (255, 0, 255), thickness=5, lineType=cv2.LINE_8)
plt.imshow(imageRectangle[:,:,::-1])
plt.show()

# to add text use .putText()
# cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16th, 1969"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0,255,0)
fontThickness = 2
cv2.putText(imageText, text, (200,700), fontFace, fontScale,
            fontColor, fontThickness, cv2.LINE_AA)
plt.imshow(imageText[:,:,::-1])
plt.show()


imageAll = imageText.copy()
cv2.rectangle(imageAll, (500, 100), (700, 600),
              (255, 0, 255), thickness=5, lineType=cv2.LINE_8)
cv2.circle(imageAll, (900,500), 100, (0,0,255),
           thickness=5, lineType=cv2.LINE_AA)
cv2.line(imageAll, (200,100), (400,100), (255,0,255),
         thickness=5, lineType=cv2.LINE_AA)
plt.imshow(imageAll[:,:,::-1])
plt.show()
