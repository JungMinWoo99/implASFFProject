from constant import *
import util.DirectoryUtils as DirectoryUtils
import cv2
from ASFFNet import *
from DataSet import ASFFDataSet


def tensor_to_img_mat(tensor):
    # 텐서를 numpy 배열로 변환하고 데이터 타입을 uint8로 변환
    img_out = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # RGB를 BGR로 변환 (cv2는 BGR을 사용)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    return img_out

asffnetG = ASFFNet().to(default_device)

test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")
asffnet_checkpoint_path = DirectoryUtils.select_file("asffnet checkpoint path")

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)
checkpoint = torch.load(asffnet_checkpoint_path)

print(checkpoint['g_loss'])

asffnetG.load_state_dict(checkpoint['gen_state_dict'])

with torch.no_grad():
    while True:
        case_num = int(input('실행할 케이스 번호를 입력하세요. '))
        data = asff_test_data[case_num]
        I_h = asffnetG(data['lp_img_tensor'].unsqueeze(0), data['g_img_tensor'].unsqueeze(0),
                       data['lp_land_bin_img_tensor'].unsqueeze(0),
                       data['lp_landmarks_tensor'].unsqueeze(0), data['g_img_landmarks_tensor'].unsqueeze(0))
        lp_img = tensor_to_img_mat(tensor_to_img(data['lp_img_tensor']))
        g_img = tensor_to_img_mat(tensor_to_img(data['g_img_tensor']))
        hp_img = tensor_to_img_mat(tensor_to_img(data['hp_img_tensor']))
        recon_img = tensor_to_img_mat(tensor_to_img(I_h.squeeze(0)))

        cv2.imshow("lp_img", lp_img)
        cv2.imshow("g_img", g_img)
        cv2.imshow("hp_img", hp_img)
        cv2.imshow("recon_img", recon_img)

        cv2.waitKey()
