from constant import *
import util.DirectoryUtils as DirectoryUtils
import cv2
from ASFFNet import *
from DataSet import ASFFDataSet
import os


def tensor_to_img_mat(tensor):
    # 텐서를 numpy 배열로 변환하고 데이터 타입을 uint8로 변환
    img_out = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # RGB를 BGR로 변환 (cv2는 BGR을 사용)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    return img_out

asffnetG = ASFFNet().to(default_device)

test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

with torch.no_grad():
    case_num = int(input('실행할 케이스 번호를 입력하세요. '))
    data = asff_test_data[case_num]

    idx = 0
    asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(idx + 1)
    while os.path.exists(asffnet_checkpoint_path):
        checkpoint = torch.load(asffnet_checkpoint_path)

        print(checkpoint['g_loss'])

        asffnetG.load_state_dict(checkpoint['gen_state_dict'])

        I_h = asffnetG(data['lp_img_tensor'].unsqueeze(0), data['g_img_tensor'].unsqueeze(0),
                       data['lp_land_bin_img_tensor'].unsqueeze(0),
                       data['lp_landmarks_tensor'].unsqueeze(0), data['g_img_landmarks_tensor'].unsqueeze(0))
        lp_img = tensor_to_img_mat(tensor_to_img(data['lp_img_tensor']))
        g_img = tensor_to_img_mat(tensor_to_img(data['g_img_tensor']))
        hp_img = tensor_to_img_mat(tensor_to_img(data['hp_img_tensor']))
        recon_img = tensor_to_img_mat(tensor_to_img(I_h.squeeze(0)))

        h_concat_img = cv2.hconcat([lp_img, g_img, hp_img, recon_img])

        # 연결된 이미지 저장
        cv2.imwrite('merged_image{}.jpg'.format(idx), h_concat_img)

        idx += 1
        asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(idx + 1)



