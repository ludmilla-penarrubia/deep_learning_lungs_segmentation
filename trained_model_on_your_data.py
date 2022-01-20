import json
import os
import argparse
import torch
import SimpleITK as sitk
import numpy as np
from munch import Munch
from tqdm import tqdm

import matplotlib.pyplot as plt
from src.model.unet import UNet
from src.model.unet3d import UNet3D
from src.utils.compute_prediction import compute_prediction
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops_table


def main(params):

    params.results_folder = os.path.join("./results_showcase")
    os.makedirs(params.results_folder, exist_ok=True)

    # Load data
    image = sitk.ReadImage(params.input_img_path)
    print(f"spacing initial : {image.GetSpacing()} and shape {image.GetSize()}")
    # Image pre processing
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    new_spacing = [1.0,1.0,1.0]
    to_resample_size=[int(original_size[0] * original_spacing[0] / new_spacing[0]), int(original_size[1] * original_spacing[1] / new_spacing[1]),
                        int(original_size[2] * original_spacing[2] / new_spacing[2])]
    
    # Gaussian filter applied to smooth the image
    gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian_filter.SetSigma(1.0)
    smooth_image = gaussian_filter.Execute(image)
    # Resampling to change the spacing of the image to isotropic 1mm
    resample_image = sitk.Resample(image1=smooth_image, size=to_resample_size,
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkBSplineTransform,
                            outputOrigin=image.GetOrigin(),
                            outputSpacing=new_spacing,
                            outputDirection=image.GetDirection(),
                            defaultPixelValue=0,
                            outputPixelType=image.GetPixelID())
    resample_size = resample_image.GetSize()
    print(f"shape of resampled image: {resample_size}")
    # Padding of the image to fit in the closest bigger image divisible by 32
    padding = []
    for i in range(3):
        if resample_size[i] // 32 == resample_size[i]/32:
            padding.append(0)
        else:
            padding.append(1)
    upper_bound = [(resample_size[0]//32 + padding[0])*32- resample_size[0], (resample_size[1]//32 + padding[1])*32 - resample_size[1],
                (resample_size[2]//32 + padding[2])*32 - resample_size[2]]
    if sum(padding) == 0:
        resized_image = resample_image
    else:
        resize = sitk.ConstantPadImageFilter()
        resize.SetConstant(-1000)
        resize.SetPadLowerBound([0,0,0])
        resize.SetPadUpperBound(upper_bound)
        resized_image = resize.Execute(resample_image)

    resized_direction = resized_image.GetDirection()
    resized_origin = resized_image.GetOrigin()
    resized_spacing = resized_image.GetSpacing()
    print(f"shape of resized image: {resized_image.GetSize()}")

    # Image to tensor
    image_array = sitk.GetArrayFromImage(resized_image)

    image_tensor = torch.tensor(image_array).float()
    image_tensor = image_tensor.unsqueeze(axis=0).unsqueeze(axis=0)
    image_tensor = image_tensor.to(device=params.device)
    params.input_size = image_array.shape

    if params.method != "3D":
        # Create empty volume to sum predictions
        pred_sum = torch.zeros((params.input_size)).to(params.device)

        # Load UNet models and predict volumes in each direction
        for slicing in tqdm(params.pre_trained_weights_path.keys(), total=3, desc="Direction prediction"):
            model = UNet(params.n_channels, params.n_classes)
            checkpoint = torch.load(params.pre_trained_weights_path[slicing], map_location=params.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(params.device)
            model.eval()
            pred_array = compute_prediction(model, slicing, image_tensor,[resized_direction, resized_origin,resized_spacing])
            pred_sum += pred_array

        # Majority vote
        prediction = torch.where(pred_sum >= 2, 1, 0)
    else:
        # Load 3D UNet model and predict the volume
        model = UNet3D(params.n_channels, params.n_classes, num_filters=4)
        checkpoint = torch.load(params.pre_trained_weights_path_3D, map_location=params.device)
        my_dic_keys = list(checkpoint["model_state_dict"].keys())
        for key in my_dic_keys:
            if 'module' in key:
                checkpoint["model_state_dict"][key[7:]]= checkpoint["model_state_dict"][key]
                del checkpoint["model_state_dict"][key]
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(params.device)
        model.eval()
        prediction = model(image_tensor)
        prediction = torch.squeeze(prediction)  # remove batch dimension
        prediction = torch.argmax(prediction, axis=0) # retrieve mask 

    # Post processing to remove small non connected objects
    prediction = prediction.detach().cpu().numpy().astype(bool)
    label_img, nums = label(prediction, return_num=True, connectivity=2)
    try:
        props = regionprops_table(label_img, properties=("label","area"))
        min_areas = props['area'][props['area'] < int((params.input_size[2] / 15)**3)]
        if len(min_areas) > 0:
            min_comp_size = max(min_areas) + 1
            print(f"Elements kept after post-processing : {props['area'][props['area'] > min_comp_size]} in voxels")
            merge_array_ = remove_small_objects(prediction, min_comp_size, in_place=True)
        else:
            pass
    except:
        print(f"An error occured, the prediction may be empty")

    plt.imshow(prediction[150,:,:])
    plt.show()
    # Processing the prediction to convert it to original image size
    prediction = sitk.GetImageFromArray(prediction.astype(np.int8))
    prediction.SetSpacing(resized_spacing)
    prediction.SetDirection(resized_direction)
    prediction.SetOrigin(resized_origin)
    prediction_ = prediction[:resample_size[0], :resample_size[1], :resample_size[2]]
    plt.imshow(sitk.GetArrayFromImage(prediction_)[150,:,:])
    plt.show()
    prediction_image = sitk.Resample(image1=prediction_, size=original_size,
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkNearestNeighbor,
                            outputOrigin=original_origin,
                            outputSpacing=original_spacing,
                            outputDirection=original_direction,
                            defaultPixelValue=0,
                            outputPixelType=image.GetPixelID())
    plt.imshow(sitk.GetArrayFromImage(prediction_image)[250,:,:])
    plt.show()
    sitk.WriteImage(prediction_image, os.path.join("./results_showcase", params.output_img_path), useCompression=True)


if __name__ == '__main__':

    # Define parameters
    parser = argparse.ArgumentParser(description="Deep Learning based Lung Segmentation algorithm")
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to input image")
    parser.add_argument('-o', "--output", required=True, type=str, help="Path to output image")
    parser.add_argument('-w',"--weights", required=True, type=str, help="Path to model weights")
    parser.add_argument('-d', "--device", required=False, default='cpu', type=str, help="device on which to run the code, by default cpu but can be gpu")
    parser.add_argument('-m', "--method", required=False, default='3D', type=str, help="Method to pick between '3D' and 'multi-2D', (default 3D)")
    args = parser.parse_args()

    params = Munch()
    params.n_channels = 1
    params.n_classes = 2
    if args.device == "gpu":
        params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        params.device = 'cpu'
    params.input_img_path = args.input
    params.output_img_path = args.output
    params.method = args.method
    params.pre_trained_weights_path_3D = args.weights
    params.pre_trained_weights_path = {
        'axial':'./data/checkpoint_best_axial.pt',
        'coronal': './data/checkpoint_best_coronal.pt',
        'sagittal':'./data/checkpoint_best_sagittal.pt'
    }

    with torch.no_grad():
        main(params)
