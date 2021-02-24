import cv2
from PIL import Image, ImageDraw, ImageFont

import argparse
import json
from time import time
from threading import Thread

from binary import Binarizer
from object_detector import ObjectDetector

DIR = 'annotated_images'

def output(results, labels):
    """
    コンソールに結果を出力する
    """
    print("--------")
    
    for result in results:
        print(f"{labels[result['class_id']]} : {result['score']}")

def count_of(target, results, labels):
    """
    指定したラベルの数を数え上げる
    """
    count = 0
    for result in results:
        if labels[result['class_id']] == target:
            count += 1

    return count

def annotate_objects(image, results, labels, filename='sample.jpg'):
    """
    バウンディングボックスとラベルを付けた画像を生成する
    """
    draw = ImageDraw.Draw(image)
    size = image.size

    for result in results:
        ymin, xmin, ymax, xmax = result['bounding_box']
        
        xmin = int(xmin * size[0])
        ymin = int(ymin * size[1])
        xmax = int(xmax * size[0])
        ymax = int(ymax * size[1])

        draw.rectangle([xmin, ymin, xmax, ymax])
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 16)
        draw.text([xmin, ymin], f"{result['score']}: {labels[result['class_id']]}", fill=(0, 0, 0) , font=font)

    image.save(filename)

def detect(image_array, detector, number):
    start = time()

    binarizer = Binarizer('bg.jpg', 50)
    gray = binarizer.binarization(image_array)
    cv2.imwrite(f'{DIR}/bin{number}.jpg', gray)
    results = detector.detect_object(gray, 0.5)

    # BGR => RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # 画像として読み込む
    with Image.fromarray(image_array) as image:
        # 結果を出力する
        labels = detector.get_labels()
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
    """
    メイン関数
    実行時の引数としてモデルファイルのパスを受け取る
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True
    )

    parser.add_argument(
        '--label', help='File path of label file.', required=True
    )

    args = parser.parse_args()

    # カメラを接続する
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = ObjectDetector(args.model, args.label)

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
            th = Thread(target=detect, args=[image_array, detector, number])
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