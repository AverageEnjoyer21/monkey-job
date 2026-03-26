from ultralytics import YOLO
from djitellopy import Tello
import os
import logging
import cv2
import time
from collections import Counter

# === Настройки окружения ===
os.environ["OPENCV_FFMPEG_SKIP_FRAME_CHECK"] = "1"

# Отключаем логирование Tello (опционально)
logging.getLogger("djitellopy").setLevel(logging.WARNING)

FRAME_SKIP = 1  # Обрабатывать каждый N-й кадр (ускорение)

# === ИНИЦИАЛИЗАЦИЯ ДРОНА ===
fly = Tello()
fly.connect()

print(f"Заряд батареи: {fly.get_battery()}%")

# Уменьшить битрейт видео для стабильности при слабом Wi-Fi
fly.set_video_bitrate(1)  # 1 Мбит/с
print("Битрейт видео установлен на 1 Мбит/с")

# Загрузка предобученной модели YOLO
model = YOLO("best_yamki-v2.pt")

unique_tracked_ids = set()


def track_and_count(frame, storage):
    results = model.track(
        frame,
        verbose=False,
        persist=True,
        imgsz=320,
        conf=0.5,
        tracker="custom_tracker.yaml",
    )[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.int().tolist()

        # Сохранение уникальных ID и имен классов
        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results.plot()


# === ВКЛЮЧЕНИЕ ВИДЕОПОТОКА ===
fly.streamon()
print("Видеопоток включён. Ожидание первого кадра...")

frame_read = fly.get_frame_read()

# Ждём первый валидный кадр
while True:
    frame = frame_read.frame
    if frame is not None and frame.size > 0:
        print("Первый кадр получен. Запуск основного цикла.")
        break
    else:
        print("Ожидание кадра...")
        time.sleep(0.1)

# Стабилизация после запуска потока
time.sleep(1)

# === ВЗЛЁТ И ПОДГОТОВКА К ПОЛЁТУ ===
# fly.takeoff()
time.sleep(3)
print("Дрон взлетел. Начинаю полёт...")

# Глобальные переменные
last_annotated_frame = None
frame_counter = 0
start_time_total = time.time()
flight_duration = 20  # Дрон будет лететь по кругу 40 секундq
flying_circle = True

try:
    while flying_circle:
        loop_start = time.time()

        # Получение кадра
        frame = frame_read.frame
        frame_counter += 1

        if frame is None or frame.size == 0:
            print("Пустой кадр — пропуск...")
            cv2.waitKey(1)
            continue

        my_frame = cv2.resize(frame, (640, 480))

        # Обработка каждого N-го кадра
        if frame_counter % FRAME_SKIP == 0:
            # Модель с трекингом
            # results = model.track(
            #     my_frame, conf=0.3, persist=True, verbose=False, imgsz=320
            # )
            # Модель без трекинга
            # results = model(my_frame, conf=0.3, verbose=False, imgsz=320)
            annotated_frame = track_and_count(my_frame, unique_tracked_ids)
            last_annotated_frame = annotated_frame
        else:
            last_annotated_frame = my_frame  # fallback

        # Отображение
        display_frame = last_annotated_frame.copy()
        cv2.imshow("Potholes Detection", cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

        # Управление дроном: полёт)
        fly.send_rc_control(
            left_right_velocity=0,  # Движение вправо
            forward_backward_velocity=20,
            up_down_velocity=0,
            yaw_velocity=0,  # Поворот по часовой стрелке
        )

        # Ограничиваем время полёта
        if time.time() - start_time_total > flight_duration:
            print("Завершение полёта.")
            flying_circle = False

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            flying_circle = False

        # Контроль FPS
        elapsed = time.time() - loop_start
        time.sleep(max(0, 0.033 - elapsed))  # ~30 FPS

except Exception as e:
    print(f"Ошибка: {e}")
finally:
    # === ПОСАДКА И ЗАВЕРШЕНИЕ ===
    fly.send_rc_control(0, 0, 0, 0)  # Остановить движение
    time.sleep(1)
    # fly.land()
    time.sleep(3)
    fly.streamoff()
    cv2.destroyAllWindows()
    print(f"Итоговая статистика:")
    final_counts = Counter(name for _, name in unique_tracked_ids)

    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")

    print(f"Полёт завершён. Заряд батареи: {fly.get_battery()}%")
    fly.end()
