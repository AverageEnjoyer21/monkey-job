import cv2
import time
import logging
from collections import Counter
from ultralytics import YOLO
from djitellopy import Tello

# --- НАСТРОЙКИ ---
MODEL_PATH = "best_very_cool_potholes_scale.pt"
FLIGHT_DURATION = 40
TRACKER_CONFIG = "custom_tracker.yaml"


logging.getLogger("djitellopy").setLevel(logging.WARNING)


def init_drone():
    """Инициализация и взлет дрона."""
    drone = Tello()
    drone.connect()
    print(f"Заряд батареи: {drone.get_battery()}%")

    drone.set_video_bitrate(Tello.BITRATE_AUTO)
    drone.set_video_fps(Tello.FPS_15)
    drone.streamon()

    frame_read = drone.get_frame_read()

    # 1. Ждем появления хоть какого-то кадра
    while frame_read.frame is None or frame_read.frame.size == 0:
        print("Ожидание видеопотока...")
        time.sleep(0.1)

    print("Стабилизация видеопотока (3 сек)...")
    for _ in range(30):  # 3 секунды при 10 fps
        _ = frame_read.frame
        time.sleep(0.1)

    print("Видео стабильно. Взлёт!")
    drone.takeoff()

    time.sleep(3)

    return drone, frame_read


def process_frame(frame, model, storage):
    """Детекция, трекинг и отрисовка кадра."""
    img = cv2.resize(frame, (640, 480))

    results = model.track(
        img, persist=True, conf=0.6, tracker=TRACKER_CONFIG, verbose=False
    )[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.int().tolist()
        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results.plot()


def fly_logic(drone, start_time):
    """Управление движением дрона."""
    # Движение по кругу
    try:
        drone.send_rc_control(
            left_right_velocity=15,
            forward_backward_velocity=0,
            up_down_velocity=0,
            yaw_velocity=-15,
        )
    except Exception as e:
        print(f"Ошибка какая та: {e}")
        return False
    return time.time() - start_time < FLIGHT_DURATION


def shutdown(drone, storage):
    """Безопасная посадка и вывод статистики."""
    print("\nЗавершение полета...")
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    drone.streamoff()
    cv2.destroyAllWindows()

    print("--- Итоговая статистика ---")
    final_counts = Counter(name for _, name in storage)
    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")

    print(f"Остаток батареи: {drone.get_battery()}%")
    drone.end()


def main():
    model = YOLO(MODEL_PATH)
    unique_tracked_ids = set()
    drone, frame_read = init_drone()

    start_time = time.time()
    active = True

    try:
        while active:
            frame = frame_read.frame
            if frame is None:
                continue

            # 1. Обработка изображения
            annotated_frame = process_frame(frame, model, unique_tracked_ids)

            # 2. Визуализация (конвертация BGR для OpenCV)
            cv2.imshow(
                "Potholes Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            )

            # 3. Управление
            active = fly_logic(drone, start_time)

            # 4. Прерывание
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        shutdown(drone, unique_tracked_ids)


if __name__ == "__main__":
    main()
