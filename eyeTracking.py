import cv2
import numpy as np
import mediapipe as mp

## Caputure 이미지에서 눈 이미지만 받아와서 이진화 한 다음 이미지 크기가 큰 순서대로
## 정렬 하고, 그 값들 중 이미지의 센터에서 가장 가까운 box를 눈동자라고 가정해보자.
def detect_pupil(image, landmarks):
    # 눈 주변 랜드마크를 사용하여 눈 영역 추출
    eye_points = np.array([(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in landmarks])
    left = int(np.min(eye_points[:, 0]))
    right = int(np.max(eye_points[:, 0]))
    top = int(np.min(eye_points[:, 1]))
    bottom = int(np.max(eye_points[:, 1]))
    eye_image = image[top:bottom, left:right]
    eye_image = cv2.resize(eye_image, dsize=(31,10),interpolation=cv2.INTER_CUBIC)

    # 눈동자 추적 (그림자 제거 방법 사용)
    gray_eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_eye = cv2.threshold(gray_eye_image, 0, 255, cv2.THRESH_OTSU)
    thresholded_eye = (255) - thresholded_eye
    # cv2.imshow('test',thresholded_eye)
    # cv2.waitKey(10)
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 큰 덩어리를 눈동자로 간주
    max_contour = max(contours, key=cv2.contourArea)
    center = np.mean(max_contour, axis=0).flatten().astype(int)

    return center + np.array([left, top])
# MediaPipe 초기 설정
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 눈 주변 랜드마크 인덱스
left_eye_landmark_indices = [33, 246, 161, 160, 159, 158, 157, 173]
right_eye_landmark_indices = [263, 466, 388, 387, 386, 385, 384, 398]

# 이전 코드에서 정의한 detect_pupil 함수 사용

# 카메라 캡처 준비
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # 이미지 처리
    results = face_mesh.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 눈동자 및 눈 중심 찾기 및 원 그리기
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_landmark_indices]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_landmark_indices]

            left_pupil = detect_pupil(frame, left_eye_landmarks)
            right_pupil = detect_pupil(frame, right_eye_landmarks)

            # 눈 중심 찾기
            left_eye_center = np.mean([(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in left_eye_landmarks], axis=0)
            right_eye_center = np.mean([(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in right_eye_landmarks], axis=0)

            if left_pupil is not None:
                radius = int(np.linalg.norm(left_pupil - left_eye_center))
                cv2.circle(frame, tuple(left_pupil.astype(int)), radius, (0, 255, 0), 1)

            if right_pupil is not None:
                radius = int(np.linalg.norm(right_pupil - right_eye_center))
                cv2.circle(frame, tuple(right_pupil.astype(int)), radius, (0, 255, 0), 1)

    cv2.imshow('Eye Tracking', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
