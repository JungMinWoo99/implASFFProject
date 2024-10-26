import os
import cv2
import torch
import util.DirectoryUtils as DirectoryUtils
from Data.DataSet import *
from torch.utils.data import DataLoader

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=1):
    """
    이미지에 랜드마크를 그리는 함수
    image: 이미지 (numpy array)
    landmarks: [2, 68] 형태의 텐서로 각 점의 x, y 좌표가 들어있음
    color: 랜드마크 점의 색상
    radius: 랜드마크 점의 반지름
    """
    for i in range(landmarks.shape[1]):
        x, y = int(landmarks[0, i]), int(landmarks[1, i])
        cv2.circle(image, (x, y), radius, color, -1)
    return image

def process_images_with_landmarks(image_folder_path, landmarks_tensor, output_folder_path, test_img, test_img_land):
    """
    이미지에 랜드마크를 그려서 저장하는 함수
    image_folder_path: 이미지가 저장된 폴더 경로
    landmarks_tensor: [N, 2, 68] 모양의 랜드마크 좌표 텐서 (N은 이미지 개수)
    output_folder_path: 결과 이미지를 저장할 폴더 경로
    """
    landmarks_tensor.append(test_img_land)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    image_filenames = os.listdir(image_folder_path)
    image_filenames = [f for f in image_filenames if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_filenames.remove(os.path.basename(test_img))
    image_filenames.append(os.path.basename(test_img))

    for idx, filename in enumerate(image_filenames):
        image_path = os.path.join(image_folder_path, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        if image is None:
            print(f"이미지 {filename}를 불러오는데 실패했습니다.")
            continue

        landmarks = landmarks_tensor[idx]  # 해당 이미지의 랜드마크
        if landmarks.shape == (2, 68):
            image_with_landmarks = draw_landmarks(image, landmarks)
            output_image_path = os.path.join(output_folder_path, filename)
            cv2.imwrite(output_image_path, image_with_landmarks)
        else:
            print(f"{filename}의 랜드마크 모양이 올바르지 않습니다. 건너뜁니다.")

# 사용 예시
test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")

wls = WLS()
wls.w = torch.load(wls_weight_path)['wls_weight']

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
wls_test_data = WLSDataSet(test_data_list)


def custom_collate_fn(batch):
    return batch[0]  # 배치 차원을 제거


test_dataloader = DataLoader(
    wls_test_data,  # 위에서 생성한 데이터 셋
    batch_size=1,
    shuffle=False,  # 데이터들의 순서는 섞어서 분할
    collate_fn=custom_collate_fn
)

while True:
    test_case = int(input("실행할 테스트 번호"))
    print(test_data_list[test_case])

    test_img = test_data_list[test_case][0]
    image_folder = test_data_list[test_case][1]
    test_img_land = wls_test_data[test_case][0]
    landmarks_tensor = wls_test_data[test_case][1]  # 임시 랜드마크 텐서, 실제 데이터로 대체 필요
    output_folder = "./tem/landmarkimg"  # 결과를 저장할 폴더 경로

    process_images_with_landmarks(image_folder, landmarks_tensor, output_folder,test_img, test_img_land)