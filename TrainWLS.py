from DataSet import WLSDataSet
from util import DirectoryUtils
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.WLS import WLS
import torch.optim as optim
import torch
from util.PrintTrainLog import print_wls_log
import os.path

train_data_set_path = DirectoryUtils.select_file("train data list csv")
test_data_set_path = DirectoryUtils.select_file("test data list csv")

file_num = 0
wls = WLS()
load_checkpoint = DirectoryUtils.ask_load_file("load checkpoint?")
if load_checkpoint:
    wls_checkpoint_path = 'wls_checkpoint{}.pth'.format(file_num)
    while os.path.exists(wls_checkpoint_path):
        file_num += 1
        wls.w = torch.load(wls_checkpoint_path)['wls_weight']
        wls_checkpoint_path = 'wls_checkpoint{}.pth'.format(file_num)

train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
wls_train_data = WLSDataSet(train_data_list)
wls_test_data = WLSDataSet(test_data_list)


def custom_collate_fn(batch):
    return batch[0]  # 배치 차원을 제거


train_dataloader = DataLoader(
    wls_train_data,  # 위에서 생성한 데이터 셋
    batch_size=1,
    shuffle=True,  # 데이터들의 순서는 섞어서 분할
    collate_fn=custom_collate_fn
)

test_dataloader = DataLoader(
    wls_test_data,  # 위에서 생성한 데이터 셋
    batch_size=1,
    shuffle=True,  # 데이터들의 순서는 섞어서 분할
    collate_fn=custom_collate_fn
)

optimizer = optim.Adam([wls.w], lr=0.0001)

loss_list = []
epoch = 50
for e in range(epoch):
    for data in tqdm(train_dataloader, desc='Processing Batches'):
        optimizer.zero_grad()
        loss, answer = wls.compute_loss(data[0], data[1])
        loss.backward()
        optimizer.step()
    loss_sum = 0
    for data in tqdm(test_dataloader, desc='calculate loss'):
        loss, answer = wls.compute_loss(data[0], data[1])
        loss_sum += loss.item()
    loss_avg = loss_sum / len(wls_test_data)
    print(loss_avg)
    loss_list.append(loss_avg)
    # 가중치 텐서 저장
    wls_checkpoint = {'wls_weight': wls.w, 'wls_loss': loss_avg}
    torch.save(wls_checkpoint, 'wls_checkpoint{}.pth'.format(file_num))
    file_num += 1

print_wls_log()
