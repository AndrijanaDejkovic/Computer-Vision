import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt


imgIn = cv.imread("coins.png")
imgGray = cv.cvtColor(imgIn, cv.COLOR_BGR2GRAY)
retval2, img = cv.threshold(imgGray, 90, 255, cv.THRESH_BINARY_INV)

cv.imshow("Input2", img)




cv.waitKey(0)
cv.destroyAllWindows()


