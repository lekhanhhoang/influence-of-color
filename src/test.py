import cv2

image = cv2.imread('test.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('scale.jpg', gray_image)