# modules/gaze/gaze_thread_example.py

import cv2
import threading
import queue
from modules.gaze.gaze_module import GazeTracker
from modules.shared_flags import RUNNING

# 🔥 camera_manager에서 공통 프레임 가져오기
from modules.camera.camera_manager import shared_frame_queue

# 분석 결과 → main.py
gaze_result_queue = queue.Queue(maxsize=5)


# ======================================================
# 👁 시선 분석 스레드 (카메라 공유 버전)
# ======================================================
def gaze_worker():
    tracker = GazeTracker()
    print("👁 Gaze Thread Started")

    while RUNNING:

        # 🔥 카메라 프레임이 없으면 잠시 대기
        if shared_frame_queue.empty():
            continue

        # 🔥 공통 카메라 프레임 가져오기
        frame = shared_frame_queue.get()

        processed = tracker.process_frame(frame)

        result = {
            "left_right": tracker.gaze_direction_x,
            "up_down": tracker.gaze_direction_y,
            "is_blinking": tracker.is_blinking,
            "ear": tracker.current_avg_ear,
        }

        # 최신 데이터만 유지
        if gaze_result_queue.full():
            try:
                gaze_result_queue.get_nowait()
            except:
                pass

        gaze_result_queue.put((processed, result))

    print("👁 Gaze Thread Stopped")


# ======================================================
# ▶️ 스레드 시작 함수
# ======================================================
def start_gaze_thread():
    t_gaze = threading.Thread(target=gaze_worker, daemon=True)
    t_gaze.start()
    print("🚀 gaze_thread_example 실행됨! (Camera 공유 버전)")
    return t_gaze


# ======================================================
# 🔍 단독 테스트 (필요할 때만 사용)
# ======================================================
if __name__ == "__main__":
    from modules.camera.camera_manager import start_camera_thread
    from modules.gaze.gaze_module import GazeTracker

    start_camera_thread()
    RUNNING = True

    start_gaze_thread()

    while True:
        if not gaze_result_queue.empty():
            frame, data = gaze_result_queue.get()

            print(f"시선: {data['left_right']}, {data['up_down']} / "
                  f"깜빡임: {data['is_blinking']} / EAR={data['ear']:.3f}")

            cv2.imshow("Gaze Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    RUNNING = False
    cv2.destroyAllWindows()
