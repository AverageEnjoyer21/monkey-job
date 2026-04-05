from ultralytics import YOLO
from djitellopy import Tello
import logging
import cv2
import time

# Отключаем логирование Tello (опционально)
logging.getLogger("djitellopy").setLevel(logging.WARNING)

# === ПАРАМЕТРЫ ОТСЛЕЖИВАНИЯ ===
DEAD_ZONE_X = 150  # Мёртвая зона по горизонтали (px)
DEAD_ZONE_Y = 100  # Мёртвая зона по вертикали (px)
TARGET_ID = None  # ID отслеживаемого человека (автовыбор при первом обнаружении)

# Индексы ключевых точек (формат COCO)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# === ЗАГРУЗКА МОДЕЛИ YOLOv8 POSE ===
model = YOLO("yolov8n-pose.pt")
fly = Tello()


def drone_init(fly):
    fly.connect()

    # Устанавливаем частоту видеосъёмки в 15 FPS
    fly.set_video_fps(Tello.FPS_15)

    # Устанавливаем автоматический битрейт изображения (можно поставить от 1 до 5)
    fly.set_video_bitrate(Tello.BITRATE_AUTO)

    # Заряд батареи
    print(f"Заряд батареи: {fly.get_battery()}%")

    # Остановить все движения при старте
    fly.send_rc_control(0, 0, 0, 0)

    # === ВКЛЮЧЕНИЕ ВИДЕОПОТОКА ===
    fly.streamon()
    print("Видеопоток включён. Ожидание первого кадра...")
    time.sleep(2)

    # Стабилизация после запуска потока
    time.sleep(1)
    print("Отслеживание запущено. Нажмите 'q' для выхода.")

    # Взлет
    # fly.takeoff()


def calculate_body_center(keypoints):
    """
    Вычисляет центр тела как среднее между плечами и бёдрами.
    :param keypoints: список ключевых точек [x, y] для человека
    :return: (cx, cy) — координаты центра тела
    """
    points = [
        keypoints[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    ]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(sum(xs) / 4), int(sum(ys) / 4)


def calculate_body_area(keypoints):
    """
    Вычисляет площадь четырёхугольника по ключевым точкам (плечи + бёдра).
    :param keypoints: список ключевых точек
    :return: площадь в пикселях
    """
    x1, y1 = keypoints[LEFT_SHOULDER]
    x2, y2 = keypoints[RIGHT_SHOULDER]
    x3, y3 = keypoints[RIGHT_HIP]
    x4, y4 = keypoints[LEFT_HIP]

    area = (
        abs(
            (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1)
            - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)
        )
        / 2
    )

    return int(area)


def select_target_person(keypoints_list, frame_width):
    """Выбирает самого крупного человека около центра кадра."""
    biggest_area = 0
    best_center = None

    for kpts in keypoints_list:
        if len(kpts) < 13:
            continue
        try:
            area = calculate_body_area(kpts)
            cx_body, cy_body = calculate_body_center(kpts)
            if area < 5000 or abs(cx_body - frame_width // 2) > frame_width * 0.4:
                continue
            if area > biggest_area * 1.2:
                biggest_area = area
                best_center = (cx_body, cy_body)
        except Exception:
            continue

    return best_center, biggest_area


def get_command(body_x, body_y, square, cx, cy):
    """
    Формирует команды управления дроном на основе положения тела.
    :param body_x: X-координата центра тела
    :param body_y: Y-координата центра тела
    :param square: площадь области тела (приближение расстояния)
    :param cx: центр кадра по X
    :param cy: центр кадра по Y
    """
    if body_x > cx + DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, 60)  # Поворот направо
        return "GO RIGHT"
    elif body_x < cx - DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, -60)  # Поворот налево
        return "GO LEFT"
    elif body_y > cy + DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, -60, 0)  # Вниз
        return "GO DOWN"
    elif body_y < cy - DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, 60, 0)  # Вверх
        return "GO UP"
    elif square > 30000:
        fly.send_rc_control(0, -40, 0, 0)  # Назад (если человек близко)
        return "GO BACK"
    elif square < 7000:
        fly.send_rc_control(0, 40, 0, 0)  # Вперёд (если далеко)
        return "GO FORWARD"
    else:
        fly.send_rc_control(0, 0, 0, 0)  # Стоп
        return "STOP"


def annotate_frame(frame, center, area, command, frame_center):
    """Добавляет визуализацию на кадр."""
    h, w = frame.shape[:2]

    # Рисуем центр кадра
    cv2.circle(frame, frame_center, 8, (0, 255, 0), -1)

    # Если человек найден
    if center is not None:
        cv2.circle(frame, center, 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "Body Center",
            (center[0] + 15, center[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        # Отображение площади в кадре
        cv2.putText(
            frame,
            f"Body Area: {area}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    # Команда всегда отображается
    cv2.putText(
        frame, command, (w - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )


def process_frame(frame):
    """Полная обработка одного кадра: детекция → выбор цели → команда → разметка."""
    # Ресайзим кадр для модели
    my_frame = cv2.resize(frame, (640, 480))
    h, w = my_frame.shape[:2]
    frame_center = (w // 2, h // 2)

    # Детекция позы
    results = model(my_frame, verbose=False)
    result = results[0]
    keypoints_list = (
        result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
    )

    # Получаем размеченный кадр без bounding box
    annotated_frame = result.plot(boxes=False)

    # Выбираем цель
    body_center, body_area = select_target_person(keypoints_list, w)

    # Определяем команду
    command = (
        get_command(*body_center, body_area, *frame_center) if body_center else "STOP"
    )

    # Добавляем надписи в кадре
    annotate_frame(annotated_frame, body_center, body_area, command, frame_center)

    return annotated_frame, command


def main():
    drone_init(fly)
    frame_read = fly.get_frame_read()
    """Основной цикл приложения."""
    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            print("Ожидание кадра...")
            time.sleep(0.1)
            continue
        break
    print("Первый кадр получен. Запуск основного цикла.")

    time.sleep(1)
    print("Отслеживание запущено. Нажмите 'q' для выхода.")

    try:
        while True:
            frame = frame_read.frame
            # Проверка на пустой кадр
            if frame is None or frame.size == 0:
                print("Пустой кадр — пропуск...")
                continue

            annotated_frame, command = process_frame(frame)
            print(command)

            cv2.imshow(
                "Pose Tracking", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nПрерывание по Ctrl+C.")
    finally:
        cv2.destroyAllWindows()
        fly.send_rc_control(0, 0, 0, 0)  # Остановить дрон
        # fly.land()    # Посадка
        print(f"Заряд батареи: {fly.get_battery()}%")
        print("Конец полёта.")
        fly.streamoff()
        fly.end()


if __name__ == "__main__":
    main()
