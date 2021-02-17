from tflite_runtime.interpreter import Interpreter

from time import sleep, time
from PIL import Image
from threading import Thread

import cv2
import json

from detect_image import detect_object, load_labels, annotate_objects

DIR = 'annotated_images'

def output(results, labels):
    print("--------")
    
    for result in results:
        print(f"{labels[result['class_id']]} : {result['score']}")

def count_of(target, results, labels):
    count = 0
    for result in results:
        if labels[result['class_id']] == target:
            count += 1

    return count

def detect(image_array, interpreter, labels, number):
    start = time()

    # BGR => RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # 画像として読み込む
    with Image.fromarray(image_array) as image:
        _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape'] # 入力サイズを取得する
        resized_image = image.resize((input_width, input_height), Image.ANTIALIAS) # 画像を加工する

        # 推論を実行する
        results = detect_object(interpreter, resized_image, 0.4)

        # 結果を出力する
        output(results, labels)

        # 人数を数え上げる
        count = count_of('person', results, labels)
        with open('/tmp/persons.json', 'w') as f:
            json.dump({'count' : count}, f)

        end = time()
        time_delta = end - start
        print(f"時間：{time_delta}秒")

        # バウンディングボックスとラベルを付けた画像を出力する
        annotate_objects(image, results, labels, f'{DIR}/captured{number}.jpg')

                 
def main():
    # カメラを接続する
    camera = cv2.VideoCapture(0)

    # ラベルを読み込む
    labels = load_labels('model/coco_labels.txt')

    # モデルを読み込む
    interpreter = Interpreter('model/detect.tflite')
    interpreter.allocate_tensors()

    number = 1  # 保存時の名前用
    pre = time()
    while True:
        # 撮影する
        _, image_array = camera.read()

        # プレビューを表示する
        cv2.imshow('preview', image_array)

        # 物体検出を行う (スレッドを立てる)
        now = time()
        if now - pre > 3:
            th = Thread(target=detect, args=[image_array, interpreter, labels, number])
            th.start()
            pre = now
            number += 1

        # ループ内にwaitKeyは必要
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()
