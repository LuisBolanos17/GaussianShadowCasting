import os
import imageio
import numpy as np

from core.datasets.preprocess.process_spin import write_to_h5py, read_spin_data
from core.utils.skeleton_utils import *


def dilate_masks(masks, extend_iter=1, kernel_size=5):
    d_kernel = np.ones((kernel_size, kernel_size))
    dilated_masks = []

    for mask in masks:
        dilated = cv2.dilate(
            mask, 
            kernel=d_kernel,
            iterations=extend_iter
        )
        dilated_masks.append(dilated)

    return np.array(dilated_masks)


def process_perfcap_data(
    data_path, 
    subject='Weipeng_outdoor', 
    ext_scale=0.001,
    img_res=(1080, 1920), 
    bbox_res=224, 
    extend_iter=2
):

    spin_data = read_spin_data(
        os.path.join(data_path, 'MonoPerfCap', f'MonoPerfCap-{subject}.h5'),
        img_res=img_res, 
        bbox_res=bbox_res
    )
    img_paths = spin_data['img_path']

    bkgd = imageio.imread(os.path.join(data_path, 'MonoPerfCap', f'{subject}/bkgd.png'))
    imgs, masks = [], []
    for i, img_path in enumerate(img_paths):
        img_path = os.path.join(data_path, img_path)
        mask_path = img_path.replace('/images/', '/masks/')

        img = imageio.imread(img_path)
        mask = imageio.imread(mask_path)[..., None]

        mask[mask < 2] = 0
        mask[mask >= 2]= 1

        #imgs.append(mask * img + (1 - mask) * bkgd)
        imgs.append(img)
        masks.append(mask)

    masks = np.array(masks)
    sampling_masks = dilate_masks(masks, extend_iter=extend_iter)[..., None]
    cam_idxs = np.arange(len(masks))
    kp_idxs = np.arange(len(masks))

    data = {
        'imgs': np.array(imgs),
        'masks': np.array(masks),
        'sampling_masks': sampling_masks,
        'kp_idxs': kp_idxs,
        'cam_idxs': cam_idxs,
        'bkgds': bkgd[None],
        'bkgd_idxs': np.zeros((len(masks),)),
        **spin_data,
    }
    h5_name = os.path.join(data_path, 'MonoPerfCap', f'{subject}/{subject}_processed_h5py.h5')
    print(f"Writing h5 file to {h5_name}")
    write_to_h5py(h5_name ,data, img_chunk_size=16)

    
if __name__ == '__main__':
    process_perfcap_data('data/', subject='Nadia_outdoor')
