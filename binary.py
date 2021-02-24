import cv2

class Binarizer():
    def __init__(self, background_path, threshold):
        self.__background = background_path
        self.__threshold = threshold

    def binarization(self, image):
        img_back = cv2.imread(self.__background)
        img_comp = image

        img_back_gray = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)
        img_comp_gray = cv2.cvtColor(img_comp, cv2.COLOR_BGR2GRAY)

        img_comp_gray = cv2.blur(img_comp_gray, (3, 3))

        #差分画像を生成
        img_diff = cv2.absdiff(img_back_gray, img_comp_gray)

        #差分画像の２値化（モノクロ化）する。閾値は50に設定
        ret, img_bin = cv2.threshold(img_diff, 50 , 255, cv2.THRESH_BINARY_INV)

        return img_bin
        
