# from djitellopy import Tello
# import time

# fly = Tello()
# fly.connect()
# fly.enable_mission_pads()
# fly.set_mission_pad_detection_direction(0)
# print(fly.get_battery())
# fly.takeoff()
# time.sleep(6)
# fly.go_xyz_speed_yaw_mid(100, 0, 100, 40, 0, 2, 8)
# time.sleep(5)
# fly.land()
# fly.end()


#####################################===ИМПЕРАТИВНЫЙ СТИЛЬ===#####################################

from ultralytics import YOLO
from djitellopy import Tello
import logging
import cv2
import time
from collections import Counter

# Отключаем логирование Tello (опционально)
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# Загрузка предобученной модели YOLO
model = YOLO("best_yamki-v2.pt")

# === ИНИЦИАЛИЗАЦИЯ ДРОНА ===
fly = Tello()
fly.connect()

print(f"Заряд батареи: {fly.get_battery()}%")

# Настройка видео
fly.set_video_bitrate(Tello.BITRATE_AUTO)
fly.set_video_fps(Tello.FPS_15)

# Включение видеопотока
fly.streamon()
print("Видеопоток включён. Ожидание первого кадра...")

frame_read = fly.get_frame_read()

# Ждём первый кадр
while True:
    frame = frame_read.frame
    if frame is not None and frame.size > 0:
        print("Первый кадр получен.")
        break
    time.sleep(0.1)

time.sleep(1)

# === ВКЛЮЧЕНИЕ МИССИОННЫХ ПЭДОВ ===
fly.enable_mission_pads()
fly.set_mission_pad_detection_direction(0)  # 0 = вниз, 1 = вперёд, 2 = оба
print("Миссионные пэды включены. Направление: вниз.")

# === ВЗЛЁТ ===
fly.takeoff()
print("Дрон взлетел.")
time.sleep(3)

# Высота 80 см для надёжного обнаружения пэдов
fly.go_xyz_speed_mid(0, 0, 80, 30, 0, 2)  # Летим на высоту 80 см над pad 2
time.sleep(3)

# Глобальные переменные
unique_tracked_ids = set()


def track_and_count(frame, storage):
    results = model.track(
        frame,
        verbose=False,
        persist=True,
        conf=0.6,
        tracker="custom_tracker.yaml",
    )[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.int().tolist()

        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results.plot()


# === ПОЛЁТ К МИССИОННЫМ ПЭДАМ ===
try:
    # Пример маршрута: переход между пэдами
    # Предположим, что pad 2 — старт, pad 3 и 4 — цели
    print("Летим к пэду 3...")
    fly.go_xyz_speed_yaw_mid(
        100, 0, 80, 40, 0, 2, 3
    )  # x=100, y=0, z=80, скорость=40, yaw=0, текущий pad=2, цель=3
    time.sleep(3)

    # Во время зависания — продолжаем детекцию
    print("Детекция во время зависания у пэда 3...")
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            continue
        my_frame = cv2.resize(frame, (640, 480))
        annotated_frame = track_and_count(my_frame, unique_tracked_ids)
        cv2.imshow(
            "Potholes Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        time.sleep(0.1)

    print("Летим к пэду 4...")
    fly.go_xyz_speed_yaw_mid(100, 100, 80, 40, 0, 3, 4)
    time.sleep(3)

    print("Детекция у пэда 4...")
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            continue
        my_frame = cv2.resize(frame, (640, 480))
        annotated_frame = track_and_count(my_frame, unique_tracked_ids)
        cv2.imshow(
            "Potholes Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        time.sleep(0.1)

except Exception as e:
    print(f"Ошибка: {e}")
finally:
    # === ЗАВЕРШЕНИЕ ПОЛЁТА ===
    fly.send_rc_control(0, 0, 0, 0)
    print("Посадка...")
    fly.land()
    fly.disable_mission_pads()
    fly.streamoff()
    cv2.destroyAllWindows()

    # Статистика
    final_counts = Counter(name for _, name in unique_tracked_ids)
    print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")
    print(f"Полёт завершён. Заряд батареи: {fly.get_battery()}%")
    fly.end()

#####################################===ФУНКЦИОНАЛЬНЫЙ СТИЛЬ===#####################################
from ultralytics import YOLO
from djitellopy import Tello
import logging
import cv2
import time
from collections import Counter


def initialize_drone():
    """
    Подключает дрон и настраивает миссионные пэды.
    Возвращает объект дрона.
    """
    fly = Tello()
    fly.connect()
    print(f"Заряд батареи: {fly.get_battery()}%")

    # Настройка видео
    fly.set_video_bitrate(Tello.BITRATE_AUTO)
    fly.set_video_fps(Tello.FPS_15)

    # Включение миссионных пэдов
    fly.enable_mission_pads()
    fly.set_mission_pad_detection_direction(0)  # Камера вниз
    print("Миссионные пэды включены. Направление: вниз.")

    return fly


def start_video_stream(fly):
    fly.streamon()
    print("Видеопоток включён. Ожидание первого кадра...")
    frame_read = fly.get_frame_read()

    while True:
        frame = frame_read.frame
        if frame is not None and frame.size > 0:
            print("Первый кадр получен.")
            break
        time.sleep(0.1)
    time.sleep(1)
    return frame_read


def takeoff_and_hover(drone, hover_height=80, pad_id=2):
    drone.takeoff()
    print("Дрон взлетел.")
    time.sleep(3)
    drone.go_xyz_speed_mid(0, 0, hover_height, 30, 0, pad_id)
    time.sleep(3)


def track_objects(frame, model, storage):
    results = model.track(
        frame, verbose=False, persist=True, conf=0.6, tracker="custom_tracker.yaml"
    )[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.int().tolist()
        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results.plot()


def hang_and_detect(frame_read, model, storage):
    start_time = time.time()
    while time.time() - start_time < 10:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            continue
        resized_frame = cv2.resize(frame, (640, 480))
        annotated_frame = track_objects(resized_frame, model, storage)
        cv2.imshow(
            "Potholes Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False  # Выход по 'q'
        time.sleep(0.1)
    return True  # Успешно завершено


def fly_to_pad(fly, x, y, z, speed, yaw, from_pad, to_pad):
    print(f"Летим к пэду {to_pad}...")
    fly.go_xyz_speed_yaw_mid(x, y, z, speed, yaw, from_pad, to_pad)
    time.sleep(3)


def cleanup_and_land(fly):
    fly.send_rc_control(0, 0, 0, 0)
    print("Посадка...")
    fly.land()
    fly.disable_mission_pads()
    fly.streamoff()
    cv2.destroyAllWindows()


def print_statistics(storage):
    final_counts = Counter(name for _, name in storage)
    print(f"\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")
    print(f"Полёт завершён.")


if __name__ == "__main__":
    model = YOLO("best_yamki-v2.pt")
    unique_tracked_ids = set()

    try:

        # === ИНИЦИАЛИЗАЦИЯ ===
        fly = initialize_drone()
        frame_read = start_video_stream(fly)
        takeoff_and_hover(fly, hover_height=80, pad_id=2)

        # === МАРШРУТ ===
        fly_to_pad(fly, 100, 0, 80, 40, 0, 2, 3)
        hang_and_detect(frame_read, model, unique_tracked_ids)

        fly_to_pad(fly, 100, 100, 80, 40, 0, 3, 4)
        hang_and_detect(frame_read, model, unique_tracked_ids)

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        cleanup_and_land(fly)
        print_statistics(unique_tracked_ids)
        print(f"Заряд батареи: {fly.get_battery()}%")
        fly.end()
