# RGB에서 H(색상)S(채도)V(명도)로 바꿔줘야함
# 사람의 몸에 유용한 컬러모델임
# 피부는 HS만 사용
# 역투영 과정
#1. 2차원 함수 구하고  (H,s만)
#2. 신뢰도 함수(비율 히스토그램) 생성

#역투영 실습
import cv2
import numpy as np

#1. 모델 영상의 2차원 H-S 히스토그램 계산
img_m = cv2.imread('model.jpeg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
#opencv에서는 hue값이 0~180까지 표현한다.
hist_m = cv2.calcHist([hsv_m], [0,1], None, [180, 256], [0,180, 0, 256])

#2. 입력 영상의  2차원 H-S 히스토그램 계산
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
hist_i = cv2.calcHist([hsv_i], [0,1], None, [180, 256], [0, 180, 0, 256])

#3. 히스토그래 정규화 (모델, 입력 영상 이미지 사이즈에 영향을 받지 않게 하기 위해서, 값 다 더하면 1이된다. )
#해석:
# img_m.shape[0] = 높이 정보 들어가있음
# img_m.shape[0] = 너비 정보 들어가있음 두개 곱하면 전체 픽셀 개수 즉 이미지 사이즈 나오게된다.
# 히스토그램 배열을 이미지 픽셀 개수로 나눠주면 정규화가 된다.
hist_m = hist_m / (img_m.shape[0] * img_m.shape[1])
#두번째 방법 -> 이미지 사이즈로 접근
hist_i = hist_i / img_i.size
print("maximum of hist_m: % f" % hist_m.max()) #값 범위 체크:  1.0이하
print("maximum of hist_i: % f" % hist_i.max()) #값 범위 체크:  1.0이하

#4. 비율 히스토그램 계산 [0:1] 사이 값이 나오게 해야함
# 해석:
# hist_i를 넣는 이유 -> hist_i에 0의 값이 존재할수 있기 때문에 아주 작은 값인 1e-7을 넣는다.
hist_r = hist_m / (hist_i + 1e-7)
# 최대값이 1.0을 넘지 않도록 하는 함수
hist_r = np.minimum(hist_r, 1.0)
print("range of hist_r: [%.1f, %1.f]" %(hist_r.min(), hist_r.max()))

#5. 히스토그램 역투영 수행
height, width = img_i.shape[0], img_i.shape[1]  # 영상의 높이와 너비 정보
# 결과 배열에는 신뢰도 점수가 들어갈꺼기 때문에 데이터타입을 실수형으로 선언
result = np.zeros_like(img_i, dtype='float32')  # 입력 영상과 동일한 크기의 0으로 초기화 된 배열 생성
h, s, v = cv2.split(hsv_i)  # 채널 분리

for i in range(height):  # 모든 픽셀을 순회하며 처리
    for j in range(width):
        h_value = h[i, j]  # (i,j) 번째 픽셀의 hue값
        s_value = s[i, j]  # (i,j) 번째 픽셀의 saturation값
        # 비율 히스토그램인 hist_r에서 신뢰도 점수 얻어온다.
        confidence = hist_r[h_value, s_value]  # (i,j)번째 픽셀의 신뢰도 점수
        result[i,j] = confidence  # 신뢰도 점수를 결과 이미지 (i,j)번째 픽셀에 저장

#6. 이진화 수행(화소값이 임계값 0.02보다 크면 255(피부이다), 그렇지 않으면 0(피부가아니다))
# 히스토그램 역투영 결과를 담은 result안에 있는 배열은 [0,1] 사이에 있기 때문에
# 임계값을 0.02로 지정해줘서 0.02보다 크면 피부이고 아니면 아니다 라고 해야함
# 역투행 수행 결과 는 result로 담아 임계값을 0.02로 담는다.
ret, thresholded = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)
#cv2.THRESH_BINARY함수는 픽셀값이 0.02보다 크면 255, 작으면 0으로 할당

#파일명으로 파일 저장
cv2.imwrite('a.jpg', thresholded)
cv2.imshow('d', thresholded)


# 7.모폴로지 연산 적용
# 역투영이 완벽하지 않기 때문에 모폴로지 연산을 사용한다.
kernel = np.ones((13,13))
improved = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
#파일명으로 파일 저장
cv2.imwrite('morphology.jpg', improved)

