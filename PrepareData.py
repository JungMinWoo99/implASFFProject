'''
학습에 사용될 이미지 데이터들을 학습하기 위한 형태로 가공하는 파트
'''

from util.DirectoryUtils import *
from util.SaveLandmarks import *
from util.SaveLQimg import *


def save_lq_img_and_land_data():
    hq_img_directory = select_folder('고해상도 이미지가 저장된 폴더를 선택하세요.')
    hq_land_directory = select_folder('고해상도 이미지의 랜드마크를 저장할 폴더를 선택하세요.')
    lq_img_directory = select_folder('저해상도 이미지를 저장할 폴더를 선택하세요.')
    lq_land_directory = select_folder('저해상도 이미지의 랜드마크를 저장할 폴더를 선택하세요.')

    split_sub_folder(hq_img_directory)
    print("split_sub_folder done")
    delete_small_imgset(hq_img_directory)
    print("delete_small_imgset done")
    rename_folders_in_directory(hq_img_directory)
    rename_file_in_directory(hq_img_directory)
    save_lq_img(hq_img_directory, lq_img_directory)
    print("save_lq_img done")
    save_landmarks(hq_img_directory, hq_land_directory)
    save_landmarks(lq_img_directory, lq_land_directory)
    print("save_landmarks done")


def make_test_and_train_list():
    make_testcase_list()
    separate_img_list_to_train_and_test_list()


# 현재 디렉토리
if __name__ == '__main12__':
    create_new_file = ask_load_file("create lq_data?")
    if create_new_file:
        save_lq_img_and_land_data()
    create_new_file = ask_load_file("create train list, test list")
    if create_new_file:
        make_test_and_train_list()

if __name__ == '__mainqwe__':
    lq_img_directory = select_folder('저해상도 이미지를 저장할 폴더를 선택하세요.')
    lq_land_directory = select_folder('저해상도 이미지의 랜드마크를 저장할 폴더를 선택하세요.')
    save_landmarks(lq_img_directory, lq_land_directory)

if __name__ == '__main__':
    make_test_and_train_list()
