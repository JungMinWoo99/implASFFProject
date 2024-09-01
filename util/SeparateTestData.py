import random
from util import DirectoryUtils
import csv

# CSV 파일을 읽어옵니다.
input_file = DirectoryUtils.select_file()
case_list = DirectoryUtils.read_list_from_csv(input_file)
random.shuffle(case_list)
print('total size' + str(len(case_list)))

# 데이터프레임을 절반으로 나눕니다.
test_data_num = int(input('test data num:'))
df1 = case_list[:-test_data_num]  # 첫 번째 절반
df2 = case_list[-test_data_num:]  # 두 번째 절반

# 나눈 데이터를 각각 새로운 CSV 파일로 저장합니다.
# CSV 파일로 저장
with open('../train_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(df1)
with open('../test_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(df2)


print("CSV 파일이 두 개의 파일로 성공적으로 나누어졌습니다.")