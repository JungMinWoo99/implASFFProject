'''
학습에 사용될 이미지 데이터들을 학습하기 위한 형태로 가공하는 파트
'''

from util.DirectoryUtils import *
from util.SaveLandmarks import *
from util.SaveLQimg import *

# 현재 디렉토리
if __name__ == '__main__':
    hq_img_directory = select_folder('고해상도 이미지가 저장된 폴더를 선택하세요.')
    delete_small_imgset(hq_img_directory)
    split_sub_folder(hq_img_directory)
    rename_folders_in_directory(hq_img_directory)
    rename_file_in_directory(hq_img_directory)
    hq_land_directory = select_folder('고해상도 이미지의 랜드마크를 저장할 폴더를 선택하세요.')
    save_landmarks(hq_img_directory, hq_land_directory)
    lq_img_directory = select_folder('저해상도 이미지를 저장할 폴더를 선택하세요.')
    save_lq_img(hq_img_directory, lq_img_directory)
    lq_land_directory = select_folder('고해상도 이미지의 랜드마크를 저장할 폴더를 선택하세요.')
    save_landmarks(lq_img_directory, lq_land_directory)
    make_testcase_list()


if __name__ == '__main123123__':
    hq_img_directory = r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet"
    fix_filename_in_directory(hq_img_directory)

if __name__ == '__main123123123__':
    hq_img_directory = r"C:\Users\minwoo\code_depository\DataSet\ProjectDataSet\img"
    del_file(hq_img_directory)

if __name__ == '__main__':
    make_testcase_list()
