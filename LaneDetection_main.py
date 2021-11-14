import cv2
import week6_assign_answer
img = cv2.imread('week5_data/test_images/solidWhiteRight.jpg')

result = week6_assign_answer.run(img)
cv2.imshow('result', result)
cv2.waitKey(0)

#아래는 저장여부
# cv2.imwrite('result.png', result)

cv2.destroyAllWindows()

# # 동영상 테스트
# cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
#
# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break
#
#     result = pipeline.run(frame)
#
#     cv2.imshow('result', result)
#     key = cv2.waitKey(1)  # -1
#     if key == ord('x'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

