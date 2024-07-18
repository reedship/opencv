import os
import cv2
import sys
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
from base64 import b64encode
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

URL = r"https://www.dropbox.com/s/p8h7ckeo2dn1jtz/opencv_bootcamp_assets_NB6.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB6.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# READ VIDEO FROM SOURCE
source = 'race_car.mp4' # source = 0 for webcam
cap = cv2.VideoCapture(source)

# read and display one frame
ret, frame = cap.read()
plt.imshow(frame[..., ::-1]); plt.show()

# display the video file using cv2.imshow instead of plt.imshow()
# if (cap.isOpened() == False):
#     print("error")

# while (cap.isOpened()):
#     ret,frame = cap.read()
#     if ret == True:
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()

# write the video to a different file
# VideoWriter object = cv.VideoWriter(filename, fourcc, fps, frameSize )
# where
#     filename: Name of the output video file.
#     fourcc: 4-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc. List of codes can be obtained at Video Codecs by FOURCC page. FFMPEG backend with MP4 container natively uses other values as fourcc code: see ObjectType, so you may receive a warning message from OpenCV about fourcc code conversion.
#     fps: Framerate of the created video stream.
#     frameSize: Size of the video frames.

# default resolutions of the frame are obtained
# convert the resolutions to int from float
if os.system("which ffmpeg") == 256: # a return of 256 means the command returned status code 1 (error)
    print("NO FFMPEG FOUND")
    sys.exit()
else:
    print("got here")
cap = cv2.VideoCapture(source)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# use videoCapture.get(cv2.CAP_PROP_FPS) to find frame rate of video capture. If you hard code a number the video will be sped up/ slowed down to match
out_avi = cv2.VideoWriter("race_car_out.avi", cv2.VideoWriter_fourcc("M","J","P","G"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))
out_mp4 = cv2.VideoWriter("race_car_out.mp4", cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# Read frame and write to file
# we will read the frames from the race car video and write the same to the two objects we created in the previous step. release objects after the task is complete.
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out_avi.write(frame)
        out_mp4.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out_avi.release()
out_mp4.release()
