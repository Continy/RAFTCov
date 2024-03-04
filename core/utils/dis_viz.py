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
