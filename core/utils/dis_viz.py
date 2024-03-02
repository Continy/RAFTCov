import numpy as np
import matplotlib.pyplot as plt
import cv2


def visualize_disparity_map(disparity_map, save_path=None):

    if disparity_map.ndim == 3:
        disparity_map = disparity_map.squeeze(0)

    normalized_disparity = cv2.normalize(disparity_map,
                                         None,
                                         alpha=0,
                                         beta=1,
                                         norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    plt.imshow(normalized_disparity, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('img.png', bbox_inches='tight', pad_inches=0)
    plt.close()


from core.utils import Utility as Utility


def generate_dummy_data():
    sample = {
        'img0': np.random.rand(480, 640, 3).astype(np.float32),
        'intrinsic':
        [np.random.rand(3, 3).astype(np.float32) for _ in range(2)],
        'fmask':
        [np.random.rand(480, 640).astype(np.float32) for _ in range(2)],
        'disp0': [np.random.rand(480, 640).astype(np.float32)],
        'depth0': [np.random.rand(480, 640).astype(np.float32)],
    }
    return sample


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

    for key in sample.keys():
        original_shape = sample[key][0].shape
        downscaled_shape = downscaled_sample[key][0].shape
        print(
            f"{key}: Original shape {original_shape}, Downscaled shape {downscaled_shape}"
        )


test_downscale_flow()
