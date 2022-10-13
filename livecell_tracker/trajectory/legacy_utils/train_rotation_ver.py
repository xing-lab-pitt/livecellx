import math

import Augmentor
import keras
import tensorflow as tf
import legacy_utils.train_config as config
from legacy_utils.cv_analysis import *
from legacy_utils.models import *
from legacy_utils.utils import *

prediction_result_dir = "./train_pred_results_rpe_DIC"
train_DEBUG = False
selected_model_class = config.selected_model_class
saved_model_dir = "./models"

# saved_model_path = './models/rpe_basic_bg_binary_v0.hdf5'
saved_model_path = "./models/dld1.hdf5"
use_binary = False
print("model path:", saved_model_path, "use binary? (if false then edt applied):", use_binary)
augment_sample_amount = 5000

n_class = 1


def augment_images(images, masks):
    """
    images : N x w x h x channel
    masks: N x w x h x channel
    """
    n_ch = images[0].shape[2]
    images = [np.moveaxis(images[i], 2, 0) for i in range(len(images))]
    masks = [np.moveaxis(masks[i], 2, 0) for i in range(len(masks))]

    combined_imgs = [np.concatenate([images[i], masks[i]], axis=0) for i in range(len(images))]

    # print('f1:', len(images), len(masks), images[0].shape, masks[0].shape, combined_imgs.shape)
    p = Augmentor.DataPipeline(combined_imgs)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)
    p.rotate(1, max_left_rotation=10, max_right_rotation=10)
    # p.zoom_random(1, percentage_area=0.5)
    p.zoom(probability=0.9, min_factor=0.1, max_factor=3)
    p.crop_by_size(1, config.image_size, config.image_size, centre=False)
    augmented_combined_images = p.sample(augment_sample_amount)
    augmented_combined_images = np.array(augmented_combined_images)

    augmented_combined_images = np.moveaxis(augmented_combined_images, 1, 3)
    augmented_images, masks = augmented_combined_images[..., :n_ch], augmented_combined_images[..., n_ch:]

    # augmented_combined_images = [np.moveaxis(augmented_combined_images[i], 0, 2) for i in range(len(augmented_combined_images))]
    # augmented_images  = [augmented_combined_images[i][..., :n_ch] \
    #                      for i in range(len(augmented_combined_images))]
    # masks = [augmented_combined_images[i][..., n_ch:] \
    #          for i in range(len(augmented_combined_images))]
    return augmented_images, masks


def preprocess_train_data(images, masks, prob_threshold=0.3, use_binary=True):
    """
    Description -
    """
    input_masks = masks  # segmentation masks
    if not use_binary:
        input_masks = [ndi.distance_transform_edt(mask) for mask in masks]
    images, masks = augment_images(images, input_masks)
    images = normalize_images(images)
    # masks[masks > prob_threshold] = 1
    # masks[masks <= prob_threshold] = 0
    return images, masks


def train(X, Y):
    """
    Description -
    """
    print("start training...")
    make_dir(saved_model_dir, abort=False)
    # X = segment_images_to_small_pieces(X, config.image_size, config.image_size)
    # Y = segment_images_to_small_pieces(Y, config.image_size, config.image_size)
    # if X.shape[-1] != 1:
    #     X = np.expand_dims(X, axis=len(X.shape))

    # X = normalize_images(X)
    print("preprocessing training data...", flush=True)
    X, Y = preprocess_train_data(X, Y)
    print("input output dataset length:", len(X), len(Y))
    print("first dataset image shape:", X[0].shape, Y[0].shape)
    epoch = 300

    batch_size = 5
    weight_file = saved_model_path
    if selected_model_class == reg_seg:
        autoencoder = reg_seg()
    else:
        autoencoder = selected_model_class(n_class, X[0].shape, Y[0].shape)
    if os.path.exists(saved_model_path):
        print("resume training from existing weight file")
        autoencoder.load_weights(saved_model_path)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=weight_file, save_weights_only=True, monitor="val_loss", mode="max", save_best_only=True
    )
    callbacks = [model_checkpoint_callback]
    history = autoencoder.fit(
        X, Y, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.1, callbacks=callbacks
    )
    autoencoder.save_weights(weight_file)


def main_train():
    """
    Description - Main function for training data.
    Input
        img_path - (string) Folder for raw images
        gt_path  - (string) Folder for mask images
    """
    # X, Y, image_infos = read_training_data()
    # X, Y, image_infos = read_crop_data(crop_dataset_path)
    # img_path = "/mnt/data0/Ke/rpe_train_data/TRITC_segs/raw_images"
    # img_path = "./basicBgCorrected_TRITC_train/raw_images"
    # gt_path = "./basicBgCorrected_TRITC_train/masks"

    # img_path = "/mnt/data0/Ke/rpe_train_data/DIC_segs/raw_images"
    # gt_path = "/mnt/data0/Ke/rpe_train_data/DIC_segs/seg_masks"
    img_path = "/net/capricorn/home/xing/huijing/Segmentation/data/2-1-21-TrainingData/Seg/model_train/raw_img"
    gt_path = "/net/capricorn/home/xing/huijing/Segmentation/data/2-1-21-TrainingData/Seg/model_train/mask"

    X, Y = prep_train_data(img_path, gt_path)
    X, Y = np.array(X), np.array(Y)
    # print(X[0].shape, Y[0].shape)
    # for regression model only, just predict 1 class regression prob
    # if Y.shape[-1] != 1:
    #     Y = Y[..., 2]
    #     Y = Y.reshape(list(Y.shape) + [1]) # expand dim
    print("X, Y shape:", X.shape, Y.shape)

    train(X, Y)


# def predict_images(X, filenames, model, save_dir):
#     make_dir(save_dir, abort=False)
#     X = np.array(X)
#     X = segment_images_to_small_pieces(X, config.image_size, config.image_size)
#     if X.shape[-1] != 1:
#         X = np.expand_dims(X, axis=len(X.shape))
#     Y = model.predict(X)
#     for i in range(len(X)):
#         save_path = os.path.join(save_dir, filenames[i])
#         image = Y[i, ...]
#         print(sum((image[..., 0] > 0.5).flatten()))
#         image *= 255
#         save_tiff_image(image, save_path)


def predict_image(image, model):
    smaller_images, offsets = segment_image_to_small_pieces(
        image, config.image_size, config.image_size, return_offset=True
    )

    smaller_images_2, offsets_2 = segment_image_to_small_pieces(
        image, config.image_size, config.image_size, return_offset=True, start_h_offset=100, start_w_offset=0
    )
    smaller_images_3, offsets_3 = segment_image_to_small_pieces(
        image, config.image_size, config.image_size, return_offset=True, start_h_offset=0, start_w_offset=100
    )
    smaller_images_4, offsets_4 = segment_image_to_small_pieces(
        image, config.image_size, config.image_size, return_offset=True, start_h_offset=77, start_w_offset=77
    )
    smaller_images_5, offsets_5 = segment_image_to_small_pieces(
        image, config.image_size, config.image_size, return_offset=True, start_h_offset=33, start_w_offset=33
    )
    smaller_images.extend(smaller_images_2 + smaller_images_3 + smaller_images_4 + smaller_images_5)
    offsets.extend(offsets_2 + offsets_3 + offsets_4 + offsets_5)
    print("input smaller images shape:", np.array(smaller_images).shape)
    prob_maps = model.predict(np.array(smaller_images), batch_size=1)
    whole_prob_map = assemble_prob_map(
        prob_maps, offsets, image.shape[0], image.shape[1], config.image_size, config.image_size, n_class
    )
    seg_image = whole_prob_map / np.amax(whole_prob_map, axis=None) * 255
    seg_image = np.minimum(255, seg_image)
    return seg_image


def main_pred():
    make_dir(prediction_result_dir, abort=False)
    X_img_paths = [
        # crop_dataset_path,
        # os.path.join(data_dir, 'chr14_labeling_JZ/converted/293t001_xy10_crop1 - DC'),
        # os.path.join(
        #     data_dir, 'chr14_labeling_JZ/converted/293t001_xy01')
        # '/mnt/data0/Ke/rpe_train_data/dics'
        # '/mnt/data0/Ke/rpe_train_data/TRITC_segs'
        # "./basicBgCorrected_TRITC_train/raw_images"
        # '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/c1_raw_tiffs/',
        # '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/c2_raw_tiffs/',
        # '/media/weikang_data/rpe_pcna_p21_72_hr_imaging/tiff_files/c3_raw_tiffs/',
        "/mnt/data0/Ke/rpe72_fiji_output/singleTiffs_c2",
        "/mnt/data0/Ke/rpe_train_data/DIC_segs/raw_images",
    ]

    # Paths for dirs containing tiffs each with multiple slices
    multi_tif_paths = [
        # '/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse/bg_corrected_data/' ]
        #'/home/ke/bg_corrected_data/'
    ]

    images, filenames = load_images_from_dirs(X_img_paths, ext="tif", return_filename=True, limit_per_dir=5)
    # images = [image[...] for image in images]
    multi_tif_images, multi_tif_filenames = load_images_from_dirs(
        multi_tif_paths, ext="tif", return_filename=True, limit_per_dir=5
    )
    print("multi tif filenames:", multi_tif_filenames)

    # add single tif image to images list
    for i, tif in enumerate(multi_tif_images):
        assert len(tif.shape) >= 3
        for j, image in enumerate(tif):
            images.append(image)
            filenames.append(multi_tif_filenames[i].replace(".tif", ".%d.tif" % (j)))

    for i, image in enumerate(images):
        images[i] = img_transform(image)

    images = np.array(images)
    # images = normalize_images(images)
    # images = prep_dic_imgs(images)

    if selected_model_class == reg_seg:
        model = reg_seg()
    else:
        x_shape = [config.image_size, config.image_size, 1]
        y_shape = [config.image_size, config.image_size, n_class]
        model = selected_model_class(n_class, x_shape, y_shape)
    model.load_weights(saved_model_path)

    H, W = list((np.array(images[0].shape[:2]) // config.image_size) * config.image_size)

    # adjust images dimenstions to be [N, imgW, imgH, nClasses]
    # if len(images.shape) == 3:
    #     images = images.reshape(list(images.shape) + [1])
    # elif len(images.shape) > 4 or len(images.shape) < 3:
    #     assert False

    for i in range(len(images)):
        print("processing %d/%d images" % (i, len(images)))
        image = images[i]
        # image = image.reshape(image.shape[:2])

        # image = np.array(np.expand_dims(image, axis=len(image.shape)))
        filename = filenames[i]

        seg_image = predict_image(image, model)
        save_path = os.path.join(prediction_result_dir, filenames[i])
        if n_class == 1:
            seg_image = seg_image.reshape(seg_image.shape[:2])
            save_tiff_image(seg_image, save_path, mode="L")
        elif n_class == 3:
            save_tiff_image(seg_image, save_path, mode="RGB")
        else:
            assert False


if __name__ == "__main__":
    main_train()
    # config.load_model_once()
    # main_pred()
