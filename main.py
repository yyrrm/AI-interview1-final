# main.py
import time
import cv2
import numpy as np
import os

# ===============================
# 한글 출력(PIL)
# ===============================
from PIL import ImageFont, ImageDraw, Image

# ===============================
# OPENAI 질문생성
# ===============================
from modules.question.question_module import make_question

# ===============================
# 공유 RUNNING 플래그
# ===============================
from modules.shared_flags import RUNNING

# ===============================
# 단일 카메라 스레드
# ===============================
from modules.camera.camera_manager import start_camera_thread

# ===============================
# 모듈별 스레드 & 큐
# ===============================
from modules.pose.pose_thread_example import start_pose_thread, result_queue as pose_result_queue
from modules.gaze.gaze_thread_example import start_gaze_thread, gaze_result_queue
from modules.expression.expression_thread_example import start_expression_thread, expression_result_queue
from modules.hands.hand_thread_example import start_hands_thread, hands_result_queue
from modules.voice.voice_thread_example import start_voice_thread, voice_result_queue


# ===============================
# 최신 값만 가져오기
# ===============================
def drain_queue(q):
    latest = None
    while not q.empty():
        latest = q.get()
    return latest


# ===============================
# 한글 텍스트 출력 함수
# ===============================
def put_korean_text(
    img_bgr,
    text,
    x,
    y,
    font_size=28,
    color=(0, 255, 255),
    font_path=r"C:\Windows\Fonts\malgun.ttf",
):
    """
    OpenCV(BGR) 이미지에 한글 텍스트 출력 (PIL 사용)
    """
    if text is None:
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x, y), str(text), font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ===============================
# 메인 실행부
# ===============================
def main():
    # 🔥 표정모듈은 감정모델이 없어도 되도록 expression_thread에서 처리함
    emotion_detector = None

    # 🔥 반드시 이 순서로 시작
    start_camera_thread()          # 1개 카메라만 공유
    start_pose_thread()
    start_gaze_thread()
    start_expression_thread(emotion_detector)
    start_hands_thread()
    start_voice_thread()

    print("\n🚀 AI Mock Interview — Main Started (q 또는 X로 종료)\n")

    latest_pose = None
    latest_gaze = None
    latest_expr = None
    latest_hands = None
    latest_voice = None

    window_name = "AI Mock Interview - Dashboard"

    # ============================
    # 질문 생성용 상태 변수
    # ============================
    last_voice_text = None
    last_question_time = 0.0
    QUESTION_COOLDOWN = 3.0

    # ============================
    # 🔥 시작 질문
    # ============================
    latest_question = "간단히 자기소개 부탁드립니다."
    printed_start_question = False

    while True:
        # ===== 최신 데이터 수신 =====
        pose_data = drain_queue(pose_result_queue)
        gaze_data = drain_queue(gaze_result_queue)
        expr_data = drain_queue(expression_result_queue)
        hands_data = drain_queue(hands_result_queue)
        voice_data = drain_queue(voice_result_queue)

        if pose_data is not None:
            latest_pose = pose_data
        if gaze_data is not None:
            latest_gaze = gaze_data
        if expr_data is not None:
            latest_expr = expr_data
        if hands_data is not None:
            latest_hands = hands_data
        if voice_data is not None:
            latest_voice = voice_data

        # ============================
        # 대시보드 화면
        # ============================
        dashboard = np.zeros((800, 1200, 3), dtype=np.uint8)

        cv2.putText(
            dashboard,
            "AI Mock Interview Dashboard",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # ========== 자세 =============
        if latest_pose is not None:
            frame, motion, _ = latest_pose
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 20:370] = f
            cv2.putText(
                dashboard,
                f"Movement: {motion:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # ========== 시선 =============
        if latest_gaze is not None:
            frame, g = latest_gaze
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 400:750] = f
            cv2.putText(
                dashboard,
                f"Gaze: {g['left_right']} / {g['up_down']}",
                (400, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # ========== 손 =============
        if isinstance(latest_hands, np.ndarray):
            f = cv2.resize(latest_hands, (350, 250))
            dashboard[350:600, 400:750] = f

        # ========== 표정 =============
        # expression_thread가 (frame, result_data) 형태로 넣어줌
        if latest_expr is not None and isinstance(latest_expr, tuple) and len(latest_expr) == 2:
            _, emo = latest_expr
            if isinstance(emo, dict):
                dominant = emo.get("dominant") or "N/A"
                cv2.putText(
                    dashboard,
                    f"Expression: {dominant}",
                    (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                au_scores = emo.get("au_scores")
                if isinstance(au_scores, dict):
                    # 오류면 에러만 표시
                    if "error" in au_scores:
                        cv2.putText(
                            dashboard,
                            "AU Score: (error)",
                            (20, 410),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        total_score = au_scores.get("total_expression_score")
                        anxiety = au_scores.get("anxiety_score")
                        positive = au_scores.get("positive_score")
                        feedback = au_scores.get("feedback")

                        cv2.putText(
                            dashboard,
                            f"AU Score: {total_score} (Anx:{anxiety}, Pos:{positive})",
                            (20, 410),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                        )

                        # 피드백은 한글이므로 PIL로 출력
                        if feedback:
                            dashboard = put_korean_text(
                                dashboard,
                                feedback,
                                20,
                                450,
                                font_size=24,
                                color=(0, 255, 255),
                            )

        # ========== 음성 =============
        if latest_voice is not None:
            text = latest_voice.get("text") if isinstance(latest_voice, dict) else None
            safe_text = text[:50] if text else "(음성 인식 실패)"

            cv2.putText(
                dashboard,
                f"Voice: {safe_text}",
                (20, 520),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # ============================
            # OpenAI 질문 생성
            # ============================
            now = time.time()
            if text and (text != last_voice_text) and ((now - last_question_time) > QUESTION_COOLDOWN):
                last_voice_text = text
                last_question_time = now

                try:
                    latest_question = make_question(text)
                    print("🤖 생성된 질문:", latest_question)
                except Exception as e:
                    latest_question = "(질문 생성 실패)"
                    print("🔥 질문 생성 오류:", e)

        # ============================
        # 질문 표시(한글은 PIL)
        # ============================
        if latest_question:
            q = latest_question[:70]
            dashboard = put_korean_text(
                dashboard,
                f"Next Q: {q}",
                20,
                600,
                font_size=30,
                color=(0, 255, 255),
            )

        # ============================
        # 창 표시
        # ============================
        try:
            cv2.imshow(window_name, dashboard)
        except cv2.error:
            print("🔥 imshow error — window closed")
            break

        if not printed_start_question:
            print("🤖 시작 질문:", latest_question)
            printed_start_question = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🔥 Window closed by user.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔚 'q' pressed. Exiting.")
            break

        time.sleep(0.01)

    # ============================
    # 전체 스레드 종료
    # ============================
    import modules.shared_flags as flags
    flags.RUNNING = False

    cv2.destroyAllWindows()
    print("🧹 Threads stopped.")
    return


if __name__ == "__main__":
    main()