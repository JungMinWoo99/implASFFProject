from constant import *
import os
import random
import shutil
import csv
import re
import tkinter as tk
from tkinter import filedialog, messagebox


def ask_load_file(title="load file?"):
    response = messagebox.askquestion(title, title)
    if response == 'yes':
        return True
    else:
        return False


def select_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()  # 창 숨기기
    folder_path = filedialog.askdirectory(title=title)
    return folder_path


def select_file(title="Select File"):
    root = tk.Tk()
    root.withdraw()  # 창 숨기기
    file_path = filedialog.askopenfilename(title=title)
    return file_path


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
                #print(f"Renamed '{old_folder_path}' to '{new_folder_path}'")
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
                    #print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                    break


def split_sub_folder(target_directory):
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]

    for index, folder_name in enumerate(sorted(folders), start=1):
        sub_folder_dir = os.path.join(target_directory, folder_name)
        file_list = [f for f in os.listdir(sub_folder_dir) if os.path.isfile(os.path.join(sub_folder_dir, f))]

        if len(file_list) > 10:
            # 10개씩 나누기 위한 변수 설정
            for split_index in range(0, len(file_list), 10):
                new_folder = f"{sub_folder_dir}_split{split_index // 10 + 1}"
                os.makedirs(new_folder, exist_ok=True)

                # 현재 10개의 파일을 이동
                for f in file_list[split_index: split_index + 10]:
                    file_path = os.path.join(sub_folder_dir, f)
                    shutil.move(file_path, new_folder)



def make_testcase_list():
    lq_img_dir = select_folder("저해상도 이미지 폴더")
    ref_img_dir = select_folder("참조 이미지 폴더")
    hq_img_dir = select_folder("고해상도 이미지 폴더")
    lq_img_folders = [f for f in os.listdir(lq_img_dir) if os.path.isdir(os.path.join(lq_img_dir, f))]
    ref_img_folders = [f for f in os.listdir(ref_img_dir) if os.path.isdir(os.path.join(ref_img_dir, f))]
    hq_img_folders = [f for f in os.listdir(hq_img_dir) if os.path.isdir(os.path.join(hq_img_dir, f))]
    case_list = []

    if len(lq_img_folders) != len(hq_img_folders):
        print("LQ_img_dir != HQ_img_dir")
    for lq_folder_name, ref_folder_name, hq_folder_name in zip(sorted(lq_img_folders), sorted(ref_img_folders), sorted(hq_img_folders)):
        lq_sub_folder_dir = os.path.join(lq_img_dir, lq_folder_name)
        ref_sub_folder_dir = os.path.join(ref_img_dir, ref_folder_name)
        hq_sub_folder_dir = os.path.join(hq_img_dir, hq_folder_name)
        lq_file_list = [f for f in os.listdir(lq_sub_folder_dir) if os.path.isfile(os.path.join(lq_sub_folder_dir, f))]
        hq_file_list = [f for f in os.listdir(hq_sub_folder_dir) if os.path.isfile(os.path.join(hq_sub_folder_dir, f))]
        if len(lq_file_list) != len(hq_file_list):
            print("lq_file_list != hq_file_list")
            print(lq_sub_folder_dir)
            print(hq_sub_folder_dir)
        for lq_f, hq_f in zip(lq_file_list, hq_file_list):
            lq_file_path = os.path.join(lq_sub_folder_dir, lq_f)
            hq_file_path = os.path.join(hq_sub_folder_dir, hq_f)
            case_list.append((lq_file_path, ref_sub_folder_dir, hq_file_path))

    # CSV 파일로 저장
    with open(img_list_csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(case_list)

def separate_img_list_to_train_and_test_list():
    # CSV 파일을 읽어옵니다.
    input_file = select_file("Select img list file")
    case_list = read_list_from_csv(input_file)
    print('total size' + str(len(case_list)))

    # 데이터프레임을 절반으로 나눕니다.
    test_data_num = int(input('test data num:'))
    df1 = case_list[:-test_data_num]  # 첫 번째 절반
    df2 = case_list[-test_data_num:]  # 두 번째 절반

    # 나눈 데이터를 각각 새로운 CSV 파일로 저장합니다.
    # CSV 파일로 저장
    with open('../train_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        random.shuffle(df1)
        writer.writerows(df1)
    with open('../test_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        random.shuffle(df2)
        writer.writerows(df2)

    print("CSV 파일이 두 개의 파일로 성공적으로 나누어졌습니다.")


def read_list_from_csv(csv_file_path):
    img_list = []
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            img_list.append(tuple(row))  # 행을 튜플로 변환하여 리스트에 추가
    return img_list


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


# 디렉토리 내 모든 파일의 이름을 수정하는 함수
def fix_filename_in_directory(target_directory):
    def fix_filename(file_name):
        # 정규 표현식을 사용하여 중간에 들어간 확장자 제거
        corrected_name = re.sub(r'\.(\w+)\.npy$', r'.npy', file_name)
        return corrected_name

    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]
    for folder in folders:
        folder_path = os.path.join(target_directory, folder)
        for filename in os.listdir(folder_path):
            # 파일 경로
            file_path = os.path.join(folder_path, filename)
            # 파일 이름 수정
            new_filename = fix_filename(filename)
            new_file_path = os.path.join(folder_path, new_filename)
            # 파일 이름이 수정된 경우에만 파일 이름 변경
            if file_path != new_file_path:
                os.rename(file_path, new_file_path)
                print(f'Renamed: {file_path} to {new_file_path}')


def copy_subfolders(src, dst):
    # Ensure the destination directory exists
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Iterate over all the items in the source directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)

        # If the item is a directory, copy it
        if os.path.isdir(src_item):
            if not os.path.exists(dst_item):
                os.makedirs(dst_item)


def del_file(target_directory):
    folders = [f for f in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, f))]
    for folder in folders:
        folder_path = os.path.join(target_directory, folder)
        for filename in os.listdir(folder_path):
            # 파일 경로
            file_path = os.path.join(folder_path, filename)
            if re.search(r'\.npy$', file_path):
                os.remove(file_path)


def get_land_data_path(img_path):
    return re.sub(r'img', r'land', re.sub(r'png', r'npy', img_path))


