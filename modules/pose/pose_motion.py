import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 최근 좌표 버퍼
coords_buffer = deque(maxlen=5)
important_joints = list(range(33))

cap = cv2.VideoCapture(0)
print("🎥 실시간 움직임 분석 시작 (종료: Q 키)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    motion = 0.0
    state_kor = "인식 중..."
    state_eng = "Detecting..."

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        coords = [(landmarks[i].x, landmarks[i].y) for i in important_joints]
        coords_buffer.append(coords)

        # 프레임 간 변화량 평균 계산
        if len(coords_buffer) == coords_buffer.maxlen:
            motion = np.mean([
                np.linalg.norm(np.array(coords_buffer[i]) - np.array(coords_buffer[i - 1]))
                for i in range(1, len(coords_buffer))
            ])

            if motion > 0.02:
                state_kor = "움직임 감지됨"
                state_eng = "Motion Detected"
            else:
                state_kor = "정지 상태 유지"
                state_eng = "Stable"

            print(f"{state_kor} | 변화량: {motion:.4f}")

        # 랜드마크 시각화
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    text = f"{state_eng} | Diff: {motion:.4f}"
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # ==============================

    cv2.imshow('Pose Motion Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()