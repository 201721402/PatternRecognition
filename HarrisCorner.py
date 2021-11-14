import cv2
import numpy as np

img = cv2.imread('./shapes.png');
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

har = cv2.cornerHarris(src = gray, blockSize=3, ksize=3, k = 0.04)
print(har)
res= np.copy(img)
im = 0.01 * har.max()

res[har > im] = [255, 0,0]
res[har <= im] = [0,0,255]
res[har < 0] = [255,255,255]

a = np.hstack((img, res))
a = cv2.resize(a, None, fx=2.0, fy=2.0)
cv2.imshow('d',a)
cv2.waitKey()