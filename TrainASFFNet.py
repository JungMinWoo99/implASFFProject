import logging
from DataSet import ASFFDataSet
from util import DirectoryUtils
from constant import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm
import model.Loss as Loss
from model.ASFFDiscriminator import DCGANDiscriminator
from model.ASFFNet import ASFFNet, weights_init, tensor_to_img
from util.PrintTrainLog import print_asff_log

use_subset = True

train_logger = logging.getLogger('train_logger')
train_logger.setLevel(logging.INFO)
file_handler1 = logging.FileHandler('../train.log')
file_handler1.setFormatter(logging.Formatter('%(message)s'))
train_logger.addHandler(file_handler1)

eval_logger = logging.getLogger('eval_logger')
eval_logger.setLevel(logging.INFO)
file_handler2 = logging.FileHandler('../eval.log')
file_handler2.setFormatter(logging.Formatter('%(message)s'))
eval_logger.addHandler(file_handler2)

torch.autograd.set_detect_anomaly(True)

asffnetG = ASFFNet().to(default_device)
asffnetD = DCGANDiscriminator().to(default_device)

asffnetG.apply(weights_init)
asffnetD.apply(weights_init)

train_data_set_path = DirectoryUtils.select_file("train data list csv")
test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")

train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_train_data = ASFFDataSet(train_data_list, wls_weight_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

if use_subset:
    # 훈련 데이터 중 30%만 사용
    train_size = int(0.3 * len(asff_train_data))
    val_size = len(asff_train_data) - train_size

    # 데이터 분할
    asff_train_data, _ = random_split(asff_train_data, [train_size, val_size])

    # 테스트 데이터 중 30%만 사용
    train_size = int(0.3 * len(asff_test_data))
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

optimizerD = optim.Adam(asffnetD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(asffnetG.parameters(), lr=0.0002, betas=(0.5, 0.999))


def get_loss(data):
    F_r = asffnetG(data['lp_img_tensor'], data['g_img_tensor'], data['lp_land_bin_img_tensor'],
                   data['lp_landmarks_tensor'], data['g_img_landmarks_tensor'])
    I_h = tensor_to_img(F_r)
    I_truth = tensor_to_img(data['hp_img_tensor'])

    fake_validity = asffnetD(I_h.detach())
    real_validity = asffnetD(I_truth.detach())
    G_loss = Loss.ASFFGLoss(I_h, I_truth, fake_validity.detach())
    D_loss = Loss.ASFFDLoss(fake_validity, real_validity)
    return G_loss, D_loss


epoch = 10
for e in range(epoch):
    asffnetG.train()
    asffnetD.train()
    mem_snp_num = 1
    for data in tqdm(train_dataloader, desc='Processing Batches'):
        G_loss, D_loss = get_loss(data)

        train_logger.info("\n train loss {}".format(e) + \
                          "\nmse_loss: " + str((tradeoff_parm_mse * G_loss["mse_loss"])) + \
                          "\nperc_loss: " + str((tradeoff_parm_perc * G_loss["perc_loss"])) + \
                          "\nstyle_loss: " + str((tradeoff_parm_style * G_loss["style_loss"])) + \
                          "\nadv_loss: " + str((tradeoff_parm_adv * G_loss["adv_loss"])) + \
                          "\ntotal_loss: " + str((G_loss["total_loss"])))

        optimizerG.zero_grad()
        G_loss["total_loss"].backward()
        optimizerG.step()

        optimizerD.zero_grad()
        D_loss.backward()
        optimizerD.step()

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
            G_loss, D_loss = get_loss(data)

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
            G_loss_avg[key] = G_loss_sum_dict[key] / len(asff_test_data)
        D_loss_avg = D_loss_sum / len(asff_test_data)

        print("\nmse_loss: " + str((tradeoff_parm_mse * G_loss_avg["mse_loss"])) + \
              "\nperc_loss: " + str((tradeoff_parm_perc * G_loss_avg["perc_loss"])) + \
              "\nstyle_loss: " + str((tradeoff_parm_style * G_loss_avg["style_loss"])) + \
              "\nadv_loss: " + str((tradeoff_parm_adv * G_loss_avg["adv_loss"])) + \
              "\ntotal_loss: " + str((G_loss_avg["total_loss"])))

        # 가중치 텐서 저장
        torch.save({
            'epoch': e + 1,
            'gen_state_dict': asffnetG.state_dict(),
            'dis_state_dict': asffnetD.state_dict(),
            'gen_optimizer': optimizerG.state_dict(),
            'dis_optimizer': optimizerD.state_dict(),
            'g_loss': G_loss_avg,
            'd_loss': D_loss_avg
        }, 'asff_train_log{}.pth'.format(e + 1))

print_asff_log()
