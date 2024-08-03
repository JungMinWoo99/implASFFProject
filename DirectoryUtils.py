import os
import shutil
import csv


def rename_folders_in_directory(target_directory):
    base_name = 'n00000'
    # 폴더 목록 가져오기
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]

    # 폴더 이름을 숫자 순서대로 변경
    idx = 1
    for folder_name in sorted(folders):
        old_folder_path = os.path.join(target_directory, folder_name)
        while True:
            new_folder_name = base_name[:len(base_name) - len(str(idx))] + str(idx)
            new_folder_path = os.path.join(target_directory, new_folder_name)
            if os.path.exists(new_folder_path):
                if old_folder_path == new_folder_path:
                    break
                else:
                    idx += 1
            else:
                # 기존 폴더 이름을 숫자로 된 새 이름으로 변경
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed '{old_folder_path}' to '{new_folder_path}'")
                break


def rename_file_in_directory(target_directory):
    base_name = 'p000'
    # 폴더 목록 가져오기
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]

    # 파일 이름을 숫자 순서대로 변경
    for folder_name in sorted(folders):
        sub_folder_dir = os.path.join(target_directory, folder_name)
        file_list = [f for f in os.listdir(sub_folder_dir) if os.path.isfile(os.path.join(sub_folder_dir, f))]
        idx = 1
        for f in file_list:
            old_file_path = os.path.join(sub_folder_dir, f)
            while True:
                new_file_name = base_name[:len(base_name) - len(str(idx))] + str(idx) + '.png'
                new_file_path = os.path.join(sub_folder_dir, new_file_name)
                if os.path.exists(new_file_path):
                    if old_file_path == new_file_path:
                        break
                    else:
                        idx += 1
                else:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                    break


def split_sub_folder(target_directory):
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]

    for index, folder_name in enumerate(sorted(folders), start=1):
        sub_folder_dir = os.path.join(target_directory, folder_name)
        file_list = [f for f in os.listdir(sub_folder_dir) if os.path.isfile(os.path.join(sub_folder_dir, f))]
        file_num = len(file_list)
        if file_num > 10:
            new_folder = sub_folder_dir + "_split"
            os.makedirs(new_folder)
            for f in file_list[file_num // 2:]:
                file_path = os.path.join(sub_folder_dir, f)
                shutil.move(file_path, new_folder)


def make_img_list(target_directory):
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]
    img_list = []

    for folder_name in sorted(folders):
        sub_folder_dir = os.path.join(target_directory, folder_name)
        file_list = [f for f in os.listdir(sub_folder_dir) if os.path.isfile(os.path.join(sub_folder_dir, f))]
        for f in file_list:
            file_path = os.path.join(sub_folder_dir, f)
            img_list.append((file_path, sub_folder_dir, file_path))

    # CSV 파일로 저장
    with open('input.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(img_list)


def delete_small_imgset(target_directory):
    # 현재 디렉토리 내의 모든 폴더를 순회
    for folder_name in os.listdir(target_directory):
        folder_path = os.path.join(target_directory, folder_name)

        # 디렉토리인지 확인
        if os.path.isdir(folder_path):
            # 폴더 내의 파일 및 하위 디렉토리 수를 계산
            items = os.listdir(folder_path)
            num_files = len([item for item in items if os.path.isfile(os.path.join(folder_path, item))])

            # 파일이 4개 미만인 폴더를 삭제
            if num_files < 4:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")

    print("Deletion process completed.")


# 현재 디렉토리
if __name__ == '__main__':
    current_directory = "D:\system\Dataset\ProjectDataSet"
    split_sub_folder(current_directory)
    rename_folders_in_directory(current_directory)
    rename_file_in_directory(current_directory)
    make_img_list(current_directory)

if __name__ == '__main123__':
    current_directory = "E:\code_depository\depository_python\FSR_project\ImpASFF\\testdir"
