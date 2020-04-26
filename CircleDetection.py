import cv2
import matplotlib.pyplot as plt
import numpy as np
#make the image blur and gray
img = cv2.medianBlur(cv2.imread("Your image", 0),5)
#original image
img_s = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#Circle detection method
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=200, param2=30, minRadius=0, maxRadius=150)

if circles is not None:#if there is a circle
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        #Circle's centers
        cv2.circle(img_s, (i[0], i[1]), 1, (0, 0, 255), 2)
        #Circle's Outers
        cv2.circle(img_s, (i[0], i[1]), i[2], (0, 255, 0), 2)
#cropping the circles out of the image
masking = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

for j in circles[0, :]:
    cv2.circle(masking, (j[0], j[1]), j[2], (205, 114, 101), 2)

final_img = cv2.bitwise_or(img, img, None, mask=masking)
cv2.imshow('Final image', final_img)
cv2.imshow('Detected circles', img_s)
cv2.waitKey(0)
cv2.destroyAllWindows()
