from ultralytics import YOLO
from djitellopy import Tello
import logging
import cv2
import time
from collections import Counter

# Отключаем логирование Tello (опционально)
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# Загрузка предобученной модели YOLO
model = YOLO("best_very_cool_potholes_scale.pt")

# === ИНИЦИАЛИЗАЦИЯ ДРОНА ==;y=
fly = Tello()
fly.connect()
fly.send_rc_control(0, 0, 0, 0)

print(f"Заряд батареи: {fly.get_battery()}%")

# Установка битрейта для видео
fly.set_video_bitrate(Tello.BITRATE_AUTO)

# Уменьшить частоту кадров для стабильности
fly.set_video_fps(Tello.FPS_15)

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
fly.takeoff()
time.sleep(3)
print("Дрон взлетел. Начинаю полёт...")

# Глобальные переменные
start_time_total = time.time()
flight_duration = 40  # Дрон будет лететь по кругу 40 секундq
flying_circle = True
unique_tracked_ids = set()

# Настройки ПИД-регулятора
# Kp = 0.15
# Kd = 0.05
# Ki = 0.01
# Ui = 0

# error_old = 0


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

        # Сохранение уникальных ID и имен классов
        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results, results.plot()


try:
    while flying_circle:
        # Получение кадра
        frame = frame_read.frame

        # Проверка на пустой кадр
        if frame is None or frame.size == 0:
            print("Пустой кадр — пропуск...")
            cv2.waitKey(1)
            continue

        # Масштабирование кадра
        my_frame = cv2.resize(frame, (640, 480))

        # Модель с трекингом
        # results = model.track(
        #     my_frame, conf=0.3, persist=True, verbose=False, imgsz=320
        # )

        # Обработка кадра с трекингом
        results, annotated_frame = track_and_count(my_frame, unique_tracked_ids)

        # Крутой пид-регулятор (возможно рабочийп)
        # if results.boxes.xywh is not None and len(results.boxes.xywh) > 0:
        #     x_center = results.boxes.xywh[0][0].item()
        #     error = x_center - 320

        #     Up = Kp * error
        #     Ud = Kd * (error - error_old)
        #     Ui = Ui + Ki * error

        #     Ui = max(min(Ui, 10), -10)
        #     U = Up + Ui + Ud
        #     speed_LR = int(max(min(U, 35), -35))
        #     error_old = error
        # else:
        #     speed_LR = 0
        #     Ui = 0
        #     error_old = 0

        # Отображение
        cv2.imshow(
            "Potholes Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )

        # Управление дроном: полёт
        fly.send_rc_control(
            left_right_velocity=25,  # Движение вправо
            forward_backward_velocity=0,
            up_down_velocity=0,
            yaw_velocity=-25,  # Поворот по часовой стрелке
        )

        # Ограничиваем время полёта
        if time.time() - start_time_total > flight_duration:
            print("Завершение полёта.")
            flying_circle = False

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            flying_circle = False

except Exception as e:
    print(f"Ошибка: {e}")
finally:
    # === ПОСАДКА И ЗАВЕРШЕНИЕ ===
    fly.send_rc_control(0, 0, 0, 0)  # Остановить движение
    fly.land()  # Посадка
    fly.streamoff()
    cv2.destroyAllWindows()
    print(f"Итоговая статистика:")
    final_counts = Counter(name for _, name in unique_tracked_ids)

    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")

    print(f"Полёт завершён. Заряд батареи: {fly.get_battery()}%")
    fly.end()
