"""
EXPERIMENTAL: some code used to generate images for presentation/report
"""


from pathlib import Path

import cv2
import numpy as np


def create_images(path):
    path = Path(path)
    image = np.ones((120, 160, 3)) * 255
    if "manyrandom" in path.as_posix():
        return image
    imgs = [cv2.imread(img_path.as_posix()) for img_path in path.iterdir()]
    imgs = [img for img in imgs if img is not None]
    if len(imgs) == 4:
        for idx, img in enumerate(imgs):
            i = idx // 2
            j = idx % 2
            image[i * 60:(i + 1) * 60, j * 80: (j + 1)*80] = cv2.resize(img, (80, 60))
    elif len(imgs) == 20:
        for idx, img in enumerate(imgs):
            i = idx // 5
            j = idx % 5
            image[i * 30:(i + 1) * 30, j * 32: (j + 1) * 32] = cv2.resize(img, (32, 30))
    elif len(imgs) == 1:
        image = cv2.resize(imgs[0], (160, 120))
    else:
        raise Exception("THIS CANNOT BE!!! i = " + str(len(imgs)))
    return image


def main():
    ### domain randomization image footage ###
    border = 2
    img_nums = [0, 5, 15]
    width = 150
    height = 150
    num_imgs = len(img_nums)
    imgs_aug = ["/home/dominik/dataspace/images/pix2pix/randbackgradscale_aug/testA/{:05d}.jpg".format(i) for i in
                img_nums]
    imgs_gt = ["/home/dominik/dataspace/images/pix2pix/randbackgradscale_aug/testB/{:05d}.jpg".format(i) for i in
               img_nums]

    image = np.ones((2 * (height + border) + border, num_imgs * (width + border) + border, 3), np.uint8) * 255
    for i in range(num_imgs):
        ver_aug = border
        ver_gt = 2 * border + height
        hor = border + i * (width + border)
        image[ver_aug:ver_aug + height, hor:hor + width, :] = cv2.resize(cv2.imread(imgs_aug[i]), (width, height))
        image[ver_gt:ver_gt + height, hor:hor + width, :] = cv2.resize(cv2.imread(imgs_gt[i]), (width, height))

    cv2.imwrite("/home/dominik/data/ETH/MasterThesis/image/domain_rand.jpg", image)

    ### neural style show nice images ###
    border = 2
    img_names = [
        "1.jpg",
        "75c30097-f184-4cd6-b32e-e1fdaf831dfc.jpg",
        "f3d7a786-ee8f-4c61-91be-618fecbf29b0.jpg"
    ]
    folders = [
        "/home/dominik/dataspace/images/camera/duckie_resized_120_160/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/style1/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/4_styles_in_batch/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/20_sib/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/20_sib_cropped/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/manyrandom/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/style3/",
        "/home/dominik/dataspace/images/neural_style/tests/120_160/20_sib_augmented/",
    ]

    image = np.ones(((len(img_names) + 1) * (120 + border) + border, len(folders) * (160 + border) + border, 3), np.uint8) * 255
    for i, folder in enumerate(folders):
        hor = border + (160 + border) * i
        if i > 0:
            folder_name = folder.split(sep="/")[-2]
            tmp_image = create_images("/home/dominik/dataspace/images/neural_style/styles/" + folder_name)
            image[border:border + 120, hor:hor + 160] = tmp_image
            cv2.imwrite("/home/dominik/data/ETH/MasterThesis/image/intermediate/{}.jpg".format(folder_name), tmp_image)
        for idx, name in enumerate(img_names):
            j = idx + 1
            ver = border + j * (120 + border)
            image[ver:ver + 120, hor:hor + 160] = cv2.imread(folder + name)

    cv2.imwrite("/home/dominik/data/ETH/MasterThesis/image/neural_style_res.jpg", image)


if __name__ == "__main__":
    main()
