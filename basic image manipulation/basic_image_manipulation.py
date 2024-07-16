import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve


def download_and_unzip(url, save_path):
    print("Downloading and extracting assets...", end="")
    urlretrieve(url, save_path)

    try:
        # use zipfile package to extract zip file from url
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])

        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/rys6f1vprily2bg/opencv_bootcamp_assets_NB2.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB2.zip")

if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# gray scale image
cb_img = cv2.imread("checkerboard_18x18.png", 0)
plt.imshow(cb_img, cmap="gray")
#print(cb_img)


# you can access individual pixels of the image
# cb_img_copy = cb_img.copy()
# cb_img_copy[2, 2] = 200
# cb_img_copy[2, 3] = 200
# cb_img_copy[3, 2] = 200
# cb_img_copy[3, 3] = 200

# # Same as above
# # cb_img_copy[2:3,2:3] = 200

# plt.imshow(cb_img_copy, cmap="gray")
# print(cb_img_copy)

# CROPPING IMAGES

# cv2 used the blue,green,red order but matplotlib expects rgb
img_NZ_bgr = cv2.imread("New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)
img_NZ_rgb = img_NZ_bgr[:,:,::-1]
plt.imshow(img_NZ_rgb)
plt.show()

cropped_region = img_NZ_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()


# RESIZING IMAGES
# parameters:
#     src:      input img
#     dsize:    output image size
#     fx:       scale for the horizontal axis (required if 'None' for dsize)
#     fy:       scale for the horizontal axis (required if 'None' for dsize)

resized_cropped_region_2x = cv2.resize( cropped_region, None, fx=2,fy=2)
plt.imshow(resized_cropped_region_2x)
plt.show()

h,w,c = cropped_region.shape # get height, width, and channels of cv image
resized_cropped_region_2x_exact = cv2.resize(cropped_region, ((2 * w), (2 * h)))
print(resized_cropped_region_2x_exact.shape)

# FLIPPING IMAGES
# .flip() parameters
# src: input image
# flipCode: 0 = flip on x-axis, 1 = flip on y-axis, -1 = flip on both
resized_cropped_region_2x_horz = cv2.flip(resized_cropped_region_2x, 1)
plt.imshow(resized_cropped_region_2x_horz)
plt.show()

resized_cropped_region_2x_vert = cv2.flip(resized_cropped_region_2x, 0)
plt.imshow(resized_cropped_region_2x_vert)
plt.show()
resized_cropped_region_2x_both = cv2.flip(resized_cropped_region_2x, -1)
plt.imshow(resized_cropped_region_2x_both)
plt.show()
