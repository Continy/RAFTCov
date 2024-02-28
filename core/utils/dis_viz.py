import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_disparity_map(disparity_map, save_path=None):
    """
    将输入的视差图(disparity_map)进行可视化，并保存为img.png。
    
    输入:
    disparity_map: (1, H, W)形状的numpy数组，代表单通道的视差图。
    
    输出:
    生成的可视化图像会被保存为当前目录下的img.png文件。
    """
    # 确保输入是(H, W)形状
    if disparity_map.ndim == 3:
        disparity_map = disparity_map.squeeze(0)

    # 归一化视差图到0-1范围内以适应伪彩色映射
    normalized_disparity = cv2.normalize(disparity_map,
                                         None,
                                         alpha=0,
                                         beta=1,
                                         norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    # 使用matplotlib的伪彩色映射进行可视化
    plt.imshow(normalized_disparity, cmap='jet')
    plt.colorbar()  # 显示色标，可选
    plt.axis('off')  # 不显示坐标轴，可选

    # 保存可视化结果
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('img.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图表，释放资源


from core.utils import Utility as Utility


def generate_dummy_data():
    sample = {
        'img0': np.random.rand(480, 640, 3).astype(np.float32),  # 假设有两个光流图
        'intrinsic':
        [np.random.rand(3, 3).astype(np.float32)
         for _ in range(2)],  # 假设有两个内参矩阵
        'fmask':
        [np.random.rand(480, 640).astype(np.float32)
         for _ in range(2)],  # 假设有两个前景掩码
        'disp0': [np.random.rand(480, 640).astype(np.float32)],  # 假设有一个视差图
        'depth0': [np.random.rand(480, 640).astype(np.float32)],  # 假设有一个深度图
    }
    return sample


# 测试 DownscaleFlow
def test_downscale_flow():
    sample = generate_dummy_data()
    downscaler = Utility.Compose([
        Utility.CropCenter((448, 640),
                           fix_ratio=False,
                           scale_w=1.0,
                           scale_disp=False),
        Utility.DownscaleFlow(4.),
        # Utility.Normalize(mean=[0.485, 0.456, 0.406],
        #                   std=[0.229, 0.224, 0.225],
        #                   rgbbgr=False,
        #                   keep_old=True),
    ])
    downscaled_sample = downscaler(sample)

    # 对比原始和缩放后的尺寸
    for key in sample.keys():
        original_shape = sample[key][0].shape
        downscaled_shape = downscaled_sample[key][0].shape
        print(
            f"{key}: Original shape {original_shape}, Downscaled shape {downscaled_shape}"
        )


# 运行测试
test_downscale_flow()
