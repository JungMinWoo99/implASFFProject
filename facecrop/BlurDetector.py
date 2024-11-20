# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import constant
from tqdm import tqdm
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def ext_landmarks(img_mat, img_path=''):
    idx = 0
    try:
        img_landmarks = FaceDetection.get_landmarks_from_image(img_mat)
    except:
        print('Error in detecting this face {}. Continue...'.format(img_path))

    if img_landmarks is None:
        print('Warning: No face is detected in {}. Continue...'.format(img_path))
        return None
    elif len(img_landmarks) > 3:
        hights = []
        for l in img_landmarks:
            hights.append(l[8, 1] - l[19, 1])  # choose the largest face
        idx = hights.index(max(hights))
        print(
            'Warning: Too many faces are detected in img, only handle the largest one...')

    selected_landmarks = img_landmarks[idx]
    return selected_landmarks


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


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(constant.g_output_img_size // 2, constant.g_output_img_size // 2))

    # 얼굴이 감지되었는지 여부 출력
    if len(faces) > 0:
        face_land = ext_landmarks(image)
        if face_land is not None:
            is_face = True
        else:
            is_face = False
    else:
        is_face = False
    return is_face


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


class NonFaceImgFilter:
    def __init__(self):
        self.img_count = 0
        self.deleted_img_count = 0

    def __call__(self, file_path):
        img = cv2.imread(file_path)
        self.img_count += 1
        if not detect_face(img):
            os.remove(file_path)
            self.deleted_img_count += 1


class GrayImgFilter:
    def __init__(self):
        self.img_count = 0
        self.deleted_img_count = 0

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

    file_paths = []

    # 모든 파일 경로 수집
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))

    # tqdm으로 진행도 표시하면서 파일 처리
    for file_path in tqdm(file_paths, desc="Processing files"):
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
    file_paths = []

    # 모든 파일 경로 수집
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))

    # tqdm으로 진행도 표시하면서 파일 처리
    for file_path in tqdm(file_paths, desc="Processing files"):
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
    find_and_process_img_file(
        start_dir="E:/code_depository/depository_python/FSR_project/ImpASFF/facecrop/CASIA-WebFace_crop", processor=d,
        extension="jpg")
    print("all: {}".format(d.img_count))
    print("del: {}".format(d.deleted_img_count))

if __name__ == '__main__':
    g = GrayImgFilter()
    find_and_process_img_file(
        start_dir="E:/code_depository/depository_python/FSR_project/ImpASFF/facecrop/CASIA-WebFace_crop_sort2",
        processor=g, extension="jpg")

if __name__ == '__main123__':
    d = BlurImgFilter()
    find_and_process_img_file(
        start_dir=r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet2\hp_img",
        processor=d, extension="png")
    print("all: {}".format(d.img_count))
    print("del: {}".format(d.deleted_img_count))

    g = GrayImgFilter()
    find_and_process_img_file(
        start_dir=r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet2\hp_img",
        processor=g, extension="png")
    print("all: {}".format(g.img_count))
    print("del: {}".format(g.deleted_img_count))
    nf = NonFaceImgFilter()
    find_and_process_img_file(
        start_dir=r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet2\hp_img",
        processor=nf, extension="png")
    print("all: {}".format(nf.img_count))
    print("del: {}".format(nf.deleted_img_count))

if __name__ == '__main__':
    nf = NonFaceImgFilter()
    find_and_process_img_file(
        start_dir=r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet2\hp_img",
        processor=nf, extension="png")
    print("all: {}".format(nf.img_count))
    print("del: {}".format(nf.deleted_img_count))