import cv2
from PIL import Image, ImageDraw, ImageFont

import argparse
import json
from time import time
from concurrent.futures import ThreadPoolExecutor

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

    annoateions = get_annotations(results, size, labels)

    for annotation in annoateions:
        ymin, xmin, ymax, xmax = annotation[0]

        draw.rectangle([xmin, ymin, xmax, ymax])
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 16)
        draw.text([xmin, ymin], f"{annotation[1]}: {annotation[2]}", fill=(0, 0, 0) , font=font)

    image.save(filename)

def get_annotations(results, size, labels):
    annotations = []
    for result in results:
        if labels[result['class_id']] != 'person':
            continue

        ymin, xmin, ymax, xmax = result['bounding_box']
        
        xmin = int(xmin * size[0])
        ymin = int(ymin * size[1])
        xmax = int(xmax * size[0])
        ymax = int(ymax * size[1])

        annotations.append([(xmin, ymin, xmax, ymax), labels[result['class_id']], result['score']])
    
    return annotations

def detect(image_array, detector, number):
    start = time()

    binarizer = Binarizer('bg.jpg', 30)
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
        # annotate_objects(image, results, labels, f'{DIR}/captured{number}.jpg')
        x, y = image_array.shape[1], image_array.shape[0]
        annotations = get_annotations(results, (x, y), labels)
        
        return annotations

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

    tpe = ThreadPoolExecutor()

    detector = ObjectDetector(args.model, args.label)

    annotations = None
    number = 1  # 保存時の名前用
    pre = time()
    while True:
        # 撮影する
        _, image_array = camera.read()

        # 物体検出を行う (スレッドを立てる)
        now = time()
        if now - pre > 2:
            annotations = tpe.submit(detect, image_array, detector, number)
            pre = now
            number += 1

        # プレビューを表示する
        if annotations != None:
            for annotation in annotations.result():
                xmin, ymin, xmax, ymax = annotation[0]
                cv2.rectangle(image_array, (xmin, ymin), (xmax, ymax), (0, 0, 255))
                cv2.putText(image_array, f'{annotation[1]}: {annotation[2]}', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        
        cv2.imshow('preview', image_array)

        # ループ内にwaitKeyは必要
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()
