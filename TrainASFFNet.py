import logging
from Data.DataSet import ASFFDataSet
from util import DirectoryUtils
from constant import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm
import model.Loss as Loss
from model.ASFFNet2 import ASFFNet, weights_init
import os.path
import util.DirectoryUtils
import model.SNGANDiscriminator as SNGAN


def print_model_size(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print('{:>8s} : {:.2f}M'.format(model_name, num_params / 1e6))


def get_d_loss(data):
    F_h, _ = asffnetG(data['lp_img_tensor'], data['g_img_tensor'], data['lp_land_bin_img_tensor'],
                   data['lp_landmarks_tensor'], data['g_img_landmarks_tensor'])
    F_truth = data['hp_img_tensor']

    fake_validity = asffnetD(F_h.detach())
    real_validity = asffnetD(F_truth)
    D_loss = Loss.ASFFadvLoss(real_validity, fake_validity)
    return D_loss


def get_g_loss(data):
    F_h, _ = asffnetG(data['lp_img_tensor'], data['g_img_tensor'], data['lp_land_bin_img_tensor'],
                   data['lp_landmarks_tensor'], data['g_img_landmarks_tensor'])
    F_truth = data['hp_img_tensor']

    fake_validity = asffnetD(F_h)
    G_loss = Loss.ASFFGLoss(F_h, F_truth, fake_validity)

    return G_loss


save_dir = './train_log'
os.makedirs(save_dir, exist_ok=True)

train_logger = logging.getLogger('train_logger')
train_logger.setLevel(logging.INFO)
file_handler1 = logging.FileHandler('./train.log')
file_handler1.setFormatter(logging.Formatter('%(message)s'))
train_logger.addHandler(file_handler1)

eval_logger = logging.getLogger('eval_logger')
eval_logger.setLevel(logging.INFO)
file_handler2 = logging.FileHandler('./eval.log')
file_handler2.setFormatter(logging.Formatter('%(message)s'))
eval_logger.addHandler(file_handler2)

use_debug_mode = DirectoryUtils.ask_load_file("use debug mode?")
torch.autograd.set_detect_anomaly(use_debug_mode)

file_num = 0
asffnetG = ASFFNet().to(default_device)
asffnetD = SNGAN.ResDiscriminator32().to(default_device)
optimizerG = optim.Adam(asffnetG.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerD = optim.Adam(asffnetD.parameters(), lr=0.0001, betas=(0.5, 0.999))

asffnetG.apply(weights_init)
# asffnetD.apply(weights_init) #SNGAN은 자체 초기화가 있음

print_model_size(asffnetG, 'asffG')
print_model_size(asffnetD, 'asffD')

load_checkpoint = DirectoryUtils.ask_load_file("load checkpoint?")
if load_checkpoint:
    asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(file_num)
    while os.path.exists(asffnet_checkpoint_path):
        file_num += 1
        asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(file_num)
    file_num -= 1
    asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(file_num)
    print('load ' + asffnet_checkpoint_path)
    checkpoint = torch.load(asffnet_checkpoint_path)
    load_g = DirectoryUtils.ask_load_file("load g_state?")
    if load_g:
        asffnetG.load_state_dict(checkpoint['gen_state_dict'])
        optimizerG.load_state_dict(checkpoint['gen_optimizer'])
    load_d = DirectoryUtils.ask_load_file("load d_state?")
    if load_d:
        asffnetD.load_state_dict(checkpoint['dis_state_dict'])
        optimizerD.load_state_dict(checkpoint['dis_optimizer'])
    file_num += 1

# train_data_set_path = DirectoryUtils.select_file("train data list csv")
train_data_set_path = "./data_csv/train_data.csv"
# test_data_set_path = DirectoryUtils.select_file("test data list csv")
test_data_set_path = "./data_csv/test_data.csv"
# wls_weight_path = DirectoryUtils.select_file("wls weight path")
wls_weight_path = "./pretrained_weight/wls_checkpoint52.pth"

train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_train_data = ASFFDataSet(train_data_list, wls_weight_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

use_subset = util.DirectoryUtils.ask_load_file("use small test set?")
if use_subset:
    # 훈련 데이터 중 5%만 사용
    train_size = int(0.05 * len(asff_train_data))
    val_size = len(asff_train_data) - train_size

    # 데이터 분할
    asff_train_data, _ = random_split(asff_train_data, [train_size, val_size])

    # 테스트 데이터 중 5%만 사용
    train_size = int(0.05 * len(asff_test_data))
    val_size = len(asff_test_data) - train_size

    # 데이터 분할
    asff_test_data, _ = random_split(asff_test_data, [train_size, val_size])

train_dataloader = DataLoader(
    asff_train_data,  # 위에서 생성한 데이터 셋
    batch_size=g_batch_size,
    shuffle=True,  # 데이터들의 순서는 섞어서 분할
)

test_dataloader = DataLoader(
    asff_test_data,  # 위에서 생성한 데이터 셋
    batch_size=g_batch_size,
    shuffle=True,  # 데이터들의 순서는 섞어서 분할
)

add_adv_loss = DirectoryUtils.ask_load_file("add adv_loss?")

epoch = 30
for e in range(epoch):
    asffnetG.train()
    asffnetD.train()

    for data in tqdm(train_dataloader, desc='Train ASFF'):
        truth_D_loss = get_d_loss(data)
        optimizerD.zero_grad()
        truth_D_loss.backward()
        optimizerD.step()

        G_loss = get_g_loss(data)
        train_logger.info("\n train loss {}".format(e) + \
                          "\nmse_loss: " + str((tradeoff_parm_mse * G_loss["mse_loss"])) + \
                          "\nperc_loss: " + str((tradeoff_parm_perc * G_loss["perc_loss"])) + \
                          "\nstyle_loss: " + str((tradeoff_parm_style * G_loss["style_loss"])) + \
                          "\nadv_loss: " + str((tradeoff_parm_adv * G_loss["adv_loss"])) + \
                          "\ntotal_loss: " + str((G_loss["total_loss"])))
        optimizerG.zero_grad()
        G_loss["total_loss"].backward()
        optimizerG.step()

    torch.cuda.empty_cache()

    with torch.no_grad():
        asffnetG.eval()
        asffnetD.eval()
        G_loss_sum_dict = {
            "mse_loss": 0,
            "perc_loss": 0,
            "style_loss": 0,
            "adv_loss": 0,
            "total_loss": 0
        }
        D_loss_sum = 0
        for data in tqdm(test_dataloader, desc='calculate loss'):
            D_loss = get_d_loss(data)
            G_loss = get_g_loss(data)

            if not add_adv_loss:
                G_loss["total_loss"] = tradeoff_parm_mse * G_loss["mse_loss"] + tradeoff_parm_perc * G_loss["perc_loss"] + tradeoff_parm_style * G_loss["style_loss"]

            for key, value in G_loss_sum_dict.items():
                G_loss_sum_dict[key] += G_loss[key].item()
            D_loss_sum += D_loss.item()

            eval_logger.info("\n eval loss {}".format(e) + \
                             "\nmse_loss: " + str((tradeoff_parm_mse * G_loss["mse_loss"])) + \
                             "\nperc_loss: " + str((tradeoff_parm_perc * G_loss["perc_loss"])) + \
                             "\nstyle_loss: " + str((tradeoff_parm_style * G_loss["style_loss"])) + \
                             "\nadv_loss: " + str((tradeoff_parm_adv * G_loss["adv_loss"])) + \
                             "\ntotal_loss: " + str((G_loss["total_loss"])))

        G_loss_avg = {
            "mse_loss": 0,
            "perc_loss": 0,
            "style_loss": 0,
            "adv_loss": 0,
            "total_loss": 0
        }
        D_loss_avg = 0

        for key, value in G_loss_sum_dict.items():
            G_loss_avg[key] = G_loss_sum_dict[key] / len(test_dataloader)
        D_loss_avg = D_loss_sum / len(test_dataloader)

        print("\nmse_loss: " + str((tradeoff_parm_mse * G_loss_avg["mse_loss"])) + \
              "\nperc_loss: " + str((tradeoff_parm_perc * G_loss_avg["perc_loss"])) + \
              "\nstyle_loss: " + str((tradeoff_parm_style * G_loss_avg["style_loss"])) + \
              "\nadv_loss: " + str((tradeoff_parm_adv * G_loss_avg["adv_loss"])) + \
              "\ntotal_loss: " + str((G_loss_avg["total_loss"])) + \
              "\nd_loss: " + str(D_loss_avg))

        log_file_path = os.path.join(save_dir, 'asff_train_log{}.pth'.format(file_num))
        torch.save({
            'epoch': file_num,
            'gen_state_dict': asffnetG.state_dict(),
            'dis_state_dict': asffnetD.state_dict(),
            'gen_optimizer': optimizerG.state_dict(),
            'dis_optimizer': optimizerD.state_dict(),
            'g_loss': G_loss_avg,
            'd_loss': D_loss_avg
        }, log_file_path)
        file_num += 1
