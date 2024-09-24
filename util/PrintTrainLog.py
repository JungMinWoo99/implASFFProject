import matplotlib.pyplot as plt
import os.path
import torch

def print_wls_log():
    idx = 0
    wls_loss_list = []
    wls_checkpoint_path = 'wls_checkpoint{}.pth'.format(idx + 1)
    while os.path.exists(wls_checkpoint_path):
        checkpoint = torch.load(wls_checkpoint_path)
        wls_loss_list.append(checkpoint['wls_loss'])
        idx += 1
        wls_checkpoint_path = 'wls_checkpoint{}.pth'.format(idx + 1)

    # 꺾은선 그래프 그리기
    plt.plot([i for i in range(1, idx + 1)], wls_loss_list, marker='o', linestyle='-')

    # 각 데이터 포인트에 값 표시
    for i, value in enumerate(wls_loss_list):
        plt.plot([i for i in range(1, idx + 1)], wls_loss_list, marker='o', linestyle='-')
        plt.annotate(f'{value:.5f}', (i, wls_loss_list[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 그래프 제목과 축 레이블
    plt.title('loss avg')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 그래프 저장
    plt.savefig('wls_loss.png')

def print_asff_log():
    idx = 0
    G_loss_list = []
    D_loss_list = []
    asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(idx + 1)
    while os.path.exists(asffnet_checkpoint_path):
        checkpoint = torch.load(asffnet_checkpoint_path)
        G_loss_list.append(checkpoint['g_loss'])
        D_loss_list.append(checkpoint['d_loss'])
        idx += 1
        asffnet_checkpoint_path = 'asff_train_log{}.pth'.format(idx + 1)

    # 각 데이터 포인트에 값 표시
    for key, _ in G_loss_list[0].items():
        loss_list = [loss[key] for loss in G_loss_list]
        plt.plot([i for i in range(1, idx + 1)], loss_list, marker='o', linestyle='-')
        for i, value in enumerate(loss_list):
            plt.annotate(f'{value:.5f}', (i, value), textcoords="offset points", xytext=(0, 10), ha='center')

        # 그래프 제목과 축 레이블
        plt.title(key)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        # 그래프 저장
        plt.savefig('{}.png'.format(key))
        plt.clf()

    # 꺾은선 그래프 그리기
    plt.plot([i for i in range(1, idx + 1)], D_loss_list, marker='o', linestyle='-')

    # 각 데이터 포인트에 값 표시
    for i, value in enumerate(D_loss_list):
        plt.plot([i for i in range(1, idx + 1)], D_loss_list, marker='o', linestyle='-')
        plt.annotate(f'{value:.5f}', (i, D_loss_list[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 그래프 제목과 축 레이블
    plt.title('d loss avg')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 그래프 저장
    plt.savefig('d_loss.png')


if __name__ == '__main__':
    print_asff_log()
