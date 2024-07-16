import cv2 as cv
img = cv.imread("/Users/reed/dev/testing/test.png")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window

for (i in Range(0,5)):
    print(i)
