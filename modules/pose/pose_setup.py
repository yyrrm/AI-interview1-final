import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Windows에서는 CAP_DSHOW 권장
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("🎥 카메라 연결 성공 — 창에서 영상 확인 가능 (q로 종료)")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ 프레임 읽기 실패")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
