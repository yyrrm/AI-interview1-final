# modules/pose/pose_thread_example.py

import cv2
import threading
import queue
from modules.pose.pose_module import PoseAnalyzer
from modules.shared_flags import RUNNING
from modules.camera.camera_manager import shared_frame_queue   # 🔥 공유 카메라 큐 사용

result_queue = queue.Queue(maxsize=5)


def pose_worker():
    analyzer = PoseAnalyzer()
    print("💪 Pose Thread Started")

    while RUNNING:
        # 카메라 프레임이 아직 생성되지 않았다면 잠시 대기
        if shared_frame_queue.empty():
            continue

        # 공통 카메라 프레임 가져오기
        frame = shared_frame_queue.get()

        processed_frame, motion, coords = analyzer.process_frame(frame)

        result = (processed_frame, motion, coords)

        # 가장 오래된 값 버리기
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except:
                pass

        result_queue.put(result)

    print("💪 Pose Thread Stopped")


def start_pose_thread():
    t_pose = threading.Thread(target=pose_worker, daemon=True)
    t_pose.start()

    print("🚀 pose_thread_example 실행됨! (Camera 공유 버전)")
    return t_pose
