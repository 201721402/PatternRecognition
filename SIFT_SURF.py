import cv2
import time

sift = cv2.xfeatures2d_SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

for scale_factor in [0.5, 1.0, 2.0, 10]:
    # 이미지 불러와서 리사이징
    img = cv2. imread('butterfly.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    print(">>", gray.shape)

    #SIFT 특징 검출 속도 계산
    t1 = time.time()
    keypoints = sift.detect(image=gray, mask=None)
    t2 = time.time()
    print("SIFT: %f sec" % (t2 - t1))

    #SURF 특징 검출 속도 계산
    t1 = time.time()
    keypoints = surf.detect(image=gray, mask=None)
    t2=time.time()
    print('SURF: %f sec' % (t2 - t1))
