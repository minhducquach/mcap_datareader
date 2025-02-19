# import required libraries
import cv2

# read the input image
img = cv2.imread('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/test_img.png')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding on the gray image to create a binary image
ret,thresh = cv2.threshold(gray,127,255,0)

# find the contours
contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# take the first contour
cnt = contours[0]
print(contours)

# compute the bounding rectangle of the contour
x,y,w,h = cv2.boundingRect(cnt)

# draw contour
img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

# draw the bounding rectangle
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# display the image with bounding rectangle drawn on it
cv2.imshow("Bounding Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()