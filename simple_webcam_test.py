# === ЭТОТ СКРИПТ НУЖЕН ДЛЯ ПРОВЕРКИ ТОГО, ЧТО ВСЕ ЗАВИСИМОСТИ УСТАНОВИЛИСЬ ПРАВИЛЬНО ===
# === ЕГО ПОЛЕЗНО ЗАПУСКАТЬ, ЧТОБЫ ПРОВЕРИТЬ, ЧТО ВСЕ УСТАНОВИЛОСЬ ПРАВИЛЬНО =====
# ===

from ultralytics import YOLO
import cv2
import time
from collections import Counter

# Загрузка предобученной модели YOLO
# Убедитесь, что файл "best_yamki-v2.pt" находится в той же папке
model = YOLO("best_yamki-v2.pt")

# === ИНИЦИАЛИЗАЦИЯ ВЕБ-КАМЕРЫ ===
# '0' — индекс встроенной камеры. Если используете внешнюю, попробуйте 1 или 2.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру.")
    exit()

print("Веб-камера запущена. Нажмите 'q' для выхода.")

# Глобальные переменные
unique_tracked_ids = set()


def track_and_count(frame, storage):
    # Запуск трекинга YOLO
    results = model.track(
        frame,
        verbose=False,
        persist=True,
        conf=0.6,
        # tracker="botsort.yaml", # Можно использовать стандартный, если custom_tracker.yaml нет
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        ids = results.boxes.id.int().tolist()
        clss = results.boxes.cls.int().tolist()

        # Сохранение уникальных ID и имен классов для статистики
        for obj_id, cls_idx in zip(ids, clss):
            storage.add((obj_id, model.names[cls_idx]))

    return results, results.plot()


try:
    while True:
        # Получение кадра с веб-камеры
        ret, frame = cap.read()

        if not ret:
            print("Ошибка чтения кадра.")
            break

        # Масштабирование кадра (как в оригинале)
        my_frame = cv2.resize(frame, (640, 480))

        # Обработка кадра с трекингом
        results, annotated_frame = track_and_count(my_frame, unique_tracked_ids)

        # Отображение результата
        # Для веб-камеры конвертация BGR2RGB обычно не нужна при cv2.imshow
        cv2.imshow("Potholes Detection - Webcam", annotated_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    # === ЗАВЕРШЕНИЕ ===
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nИтоговая статистика:")
    final_counts = Counter(name for _, name in unique_tracked_ids)

    if not final_counts:
        print("Объекты не обнаружены.")
    for obj_name, count in final_counts.items():
        print(f"- {obj_name}: {count}")

    print("Программа завершена.")
