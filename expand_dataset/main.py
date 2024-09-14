import cv2
import os
import numpy as np

# 원본 폴더와 저장할 폴더 경로 설정
input_folder = 'img'

for i in range(1, 1517):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516:05d}'  # n00002는 n01518에 저장

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)

            # 좌우 반전
            flipped_image = cv2.flip(image, 1)

            # 밝기 조절 (30% 어둡게)
            darkened_image = np.clip(flipped_image * 0.7, 0, 255).astype(np.uint8)

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), darkened_image)


# 폴더 이름 설정 및 출력 폴더 생성
for i in range(1, 1518):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516+1516:05d}'  # n00002는 n01518에 저장

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)

            # 좌우 반전
            flipped_image = cv2.flip(image, 1)

            # 밝기 조절 (30% 어둡게)
            darkened_image = np.clip(flipped_image * 0.7, 0, 255).astype(np.uint8)

            # 이미지 확대 (15% 크기 증가)
            height, width = darkened_image.shape[:2]
            new_size = (int(width * 1.15), int(height * 1.15))
            resized_image = cv2.resize(darkened_image, new_size, interpolation=cv2.INTER_LINEAR)

            # 중앙 부분 크롭
            crop_height = height
            crop_width = width
            start_x = (resized_image.shape[1] - crop_width) // 2
            start_y = (resized_image.shape[0] - crop_height) // 2
            cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # 이미지 회전 (30도)
            center = (cropped_image.shape[1] // 2, cropped_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)  # 30도 회전
            rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (cropped_image.shape[1], cropped_image.shape[0]))

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), cropped_image)

print("작업이 완료되었습니다.")

for i in range(1, 1517):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516+1516 + 1516:05d}'

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)

            # 좌우 반전
            flipped_image = cv2.flip(image, 1)

            # 이미지 회전 (15도)
            center = (flipped_image.shape[1] // 2, flipped_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)  # 15도 회전
            rotated_image = cv2.warpAffine(flipped_image, rotation_matrix, (flipped_image.shape[1], flipped_image.shape[0]))

            # 이미지 확대 (15% 크기 증가)
            height, width = rotated_image.shape[:2]
            new_size = (int(width * 1.15), int(height * 1.15))
            resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_LINEAR)

            # 중앙 부분 크롭
            crop_height = height
            crop_width = width
            start_x = (resized_image.shape[1] - crop_width) // 2
            start_y = (resized_image.shape[0] - crop_height) // 2
            cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), cropped_image)
for i in range(1, 1517):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516+1516 + 1516+1516:05d}'

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)



            # 이미지 회전 (15도)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)  # 15도 회전
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # 이미지 확대 (15% 크기 증가)
            height, width = rotated_image.shape[:2]
            new_size = (int(width * 1.15), int(height * 1.15))
            resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_LINEAR)

            # 중앙 부분 크롭
            crop_height = height
            crop_width = width
            start_x = (resized_image.shape[1] - crop_width) // 2
            start_y = (resized_image.shape[0] - crop_height) // 2
            cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), cropped_image)
for i in range(1, 1517):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516 + 1516 +1516+ 1516+1516:05d}'

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)

            # 좌우 반전
            flipped_image = cv2.flip(image, 1)

            # 이미지 회전 (-15도)
            center = (flipped_image.shape[1] // 2, flipped_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -15, 1.0)  # -15도 회전
            rotated_image = cv2.warpAffine(flipped_image, rotation_matrix, (flipped_image.shape[1], flipped_image.shape[0]))

            # 이미지 확대 (15% 크기 증가)
            height, width = rotated_image.shape[:2]
            new_size = (int(width * 1.2), int(height * 1.2))
            resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_LINEAR)

            # 중앙 부분 크롭
            crop_height = height
            crop_width = width
            start_x = (resized_image.shape[1] - crop_width) // 2
            start_y = (resized_image.shape[0] - crop_height) // 2
            cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), cropped_image)
for i in range(1, 1517):
    folder_name = f'n{i:05d}'
    output_folder = f'n{i + 1516 + 1516 +1516+ 1516+1516+1516:05d}'

    input_path = os.path.join(input_folder, folder_name)
    output_path = os.path.join(input_folder, output_folder)  # 동일한 img 폴더 내

    if not os.path.exists(input_path):
        continue

    # 출력 폴더 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 이미지 파일 처리
    for j in range(1, 8):
        image_name = f'p{j:03d}.png'
        image_path = os.path.join(input_path, image_name)

        if os.path.exists(image_path):
            # 이미지 읽기
            image = cv2.imread(image_path)


            # 이미지 회전 (-15도)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -15, 1.0)  # -15도 회전
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], flipped_image.shape[0]))

            # 이미지 확대 (15% 크기 증가)
            height, width = rotated_image.shape[:2]
            new_size = (int(width * 1.2), int(height * 1.2))
            resized_image = cv2.resize(rotated_image, new_size, interpolation=cv2.INTER_LINEAR)

            # 중앙 부분 크롭
            crop_height = height
            crop_width = width
            start_x = (resized_image.shape[1] - crop_width) // 2
            start_y = (resized_image.shape[0] - crop_height) // 2
            cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # 이미지 저장
            cv2.imwrite(os.path.join(output_path, image_name), cropped_image)

print("작업이 완료되었습니다.")










