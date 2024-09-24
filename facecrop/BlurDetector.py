# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob

def is_gray_img(image):
    # 컬러 이미지를 흑백 이미지로 변환하여 비교
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if (b == g).all() and (b == r).all():
            return True
    return False

def detect_blur_fft(image, size=60):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w, c) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = np.abs(recon)
    magnitude[magnitude == 0] = 1e-10  # 0 값을 작은 값으로 대체
    magnitude = 20 * np.log(magnitude)
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

class BlurAnalyser:
    def __init__(self):
        self.blur_val_list = []

    def __call__(self, img):
        self.blur_val_list.append(detect_blur_fft(img))


class BlurImgFilter:
    def __init__(self, threshold=53):
        self.threshold = threshold
        self.img_count = 0
        self.deleted_img_count = 0

    def __call__(self, file_path):
        img = cv2.imread(file_path)
        self.img_count += 1
        if detect_blur_fft(img) < self.threshold:
            os.remove(file_path)
            self.deleted_img_count += 1


class GrayImgFilter:
    def __call__(self, file_path):
        img = cv2.imread(file_path)
        if is_gray_img(img):
            os.remove(file_path)


def find_and_process_img(start_dir, processor, extension):
    """
    특정 디렉토리와 하위 디렉토리를 탐색하면서 특정 확장자의 파일을 찾아 엽니다.

    Args:
        start_dir (str): 탐색을 시작할 디렉토리.
        extension (str): 찾을 파일의 확장자 (예: '.txt').
        processor : 이미지를 처리를 객체
    Returns:
        None
    """
    for root, dirs, files in os.walk(start_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                processor(img)

def find_and_process_img_file(start_dir, processor, extension):
    """
    특정 디렉토리와 하위 디렉토리를 탐색하면서 특정 확장자의 파일을 찾아 엽니다.

    Args:
        start_dir (str): 탐색을 시작할 디렉토리.
        extension (str): 찾을 파일의 확장자 (예: '.txt').
        processor : 이미지를 처리를 객체
    Returns:
        None
    """
    for root, dirs, files in os.walk(start_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                print(file_path)
                processor(file_path)


if __name__ == '__main1__':
    quality_list = []
    a = BlurAnalyser()
    find_and_process_img(start_dir="D:/system/Dataset/CelebHQRefForRelease", processor=a, extension="png")
    # 히스토그램 생성
    plt.hist(a.blur_val_list, bins=10, edgecolor='black')

    # 제목과 레이블 설정
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 히스토그램 출력
    plt.show()


if __name__ == '__main2__':
    d = BlurImgFilter()
    find_and_process_img_file(start_dir="E:/code_depository/depository_python/FSR_project/ImpASFF/facecrop/CASIA-WebFace_crop", processor=d, extension="jpg")
    print("all: {}".format(d.img_count))
    print("del: {}".format(d.deleted_img_count))


if __name__ == '__main__':
    g = GrayImgFilter()
    find_and_process_img_file(start_dir="E:/code_depository/depository_python/FSR_project/ImpASFF/facecrop/CASIA-WebFace_crop_sort2", processor=g, extension="jpg")