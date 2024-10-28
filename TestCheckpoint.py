import cv2
from model.ASFFNet2 import *
from Data.DataSet import ASFFDataSet
import os
from constant import *
import util.DirectoryUtils as DirectoryUtils
import numpy as np
from util.PrintTrainLog import print_asff_log
import os.path as osp
import time
from util.VisualizeFeature import visualize_feature_map


def tensor_to_img_mat(tensor):
    # 텐서를 numpy 배열로 변환하고 데이터 타입을 uint8로 변환
    img_out = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # RGB를 BGR로 변환 (cv2는 BGR을 사용)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    return img_out


test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")
log_dir_path = DirectoryUtils.select_folder("asff log dir path")

img_out_path = './TestExamples/TestResults'
c_time = time.strftime("%m-%d_%H-%M", time.localtime())
img_save_path = osp.join(img_out_path + '_' + c_time)
os.makedirs(img_save_path, exist_ok=True)

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

print_asff_log(log_dir_path)

with torch.no_grad():
    asffnetG = ASFFNet().to(default_device)

    # 가장 작은 loss 값을 추적할 변수 초기화
    min_loss = float('inf')
    best_checkpoint_path = None

    # 폴더 내 모든 .pth 파일 중 asff_train_log로 시작하는 파일 찾기
    for filename in os.listdir(log_dir_path):
        if filename.startswith("asff_train_log") and filename.endswith(".pth"):
            checkpoint_path = os.path.join(log_dir_path, filename)

            # 체크포인트 파일 불러오기
            checkpoint = torch.load(checkpoint_path)
            loss = checkpoint['g_loss']

            # total_loss가 가장 작은 체크포인트 찾기
            if loss["total_loss"] < min_loss:
                min_loss = loss["total_loss"]
                best_checkpoint_path = checkpoint_path

    # 결과 출력
    print("Best checkpoint path:", best_checkpoint_path)
    print("Minimum total loss:", min_loss)

    checkpoint = torch.load(best_checkpoint_path)
    loss = checkpoint['g_loss']
    print("\nmse_loss: " + str((tradeoff_parm_mse * loss["mse_loss"])) + \
          "\nperc_loss: " + str((tradeoff_parm_perc * loss["perc_loss"])) + \
          "\nstyle_loss: " + str((tradeoff_parm_style * loss["style_loss"])) + \
          "\ntotal_loss: " + str((loss["total_loss"])))

    asffnetG.load_state_dict(checkpoint['gen_state_dict'])
    asffnetG.eval()

    while True:
        case_num = int(input('실행할 케이스 번호를 입력하세요. '))
        data = asff_test_data[case_num]

        I_h, fea = asffnetG(data['lp_img_tensor'].unsqueeze(0), data['g_img_tensor'].unsqueeze(0),
                            data['lp_land_bin_img_tensor'].unsqueeze(0),
                            data['lp_landmarks_tensor'].unsqueeze(0), data['g_img_landmarks_tensor'].unsqueeze(0))
        lp_img = tensor_to_img_mat(tensor_to_img(data['lp_img_tensor']))
        g_img = tensor_to_img_mat(tensor_to_img(data['g_img_tensor']))
        hp_img = tensor_to_img_mat(tensor_to_img(data['hp_img_tensor']))
        recon_img = tensor_to_img_mat(tensor_to_img(I_h.squeeze(0)))

        h_concat_img = cv2.hconcat([lp_img, g_img, hp_img, recon_img])

        for key, item in fea.items():
            visualize_feature_map(item, key)

        # 연결된 이미지 저장
        cv2.imwrite(osp.join(img_save_path, 'merged_image.jpg'), h_concat_img)

