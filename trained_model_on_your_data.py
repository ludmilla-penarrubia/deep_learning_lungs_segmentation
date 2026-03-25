
import os
import argparse
import torch
import SimpleITK as sitk
import numpy as np
from munch import Munch
from tqdm import tqdm
from time import time
import logging

import matplotlib.pyplot as plt
from src.model.unet import UNet
from src.model.unet3d import UNet3D
from src.utils.compute_prediction import compute_prediction
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops_table
from memory_profiler import profile

import resource
log = logging.getLogger(__name__)
def mem_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return f'mem usage={usage[2]/1024.0} mb'

#@profile

def main(params, log=log, logging_level=logging.INFO):
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
    params.results_folder = os.path.join("./results_showcase")
    os.makedirs(params.results_folder, exist_ok=True)

    # Load data
    start_zero = time()
    image = sitk.ReadImage(params.input_img_path)
    log.info(f"Spacing initial : {image.GetSpacing()} and shape {image.GetSize()}")
    # Image pre processing
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    new_spacing = [params.spacing, params.spacing, params.spacing]

    mask_outbound_values = image > -1000
    # Apply the mask to set all values above the upper bound to the upper bound
    clipped_image = sitk.Mask(image, mask_outbound_values, outsideValue=-1024)

    default_pixel_value = -1000
    #clipped_image = sitk.Clamp(casted_image, -1000, float('inf'))
    blurred_image = sitk.SmoothingRecursiveGaussian(clipped_image, sigma=1)
    
    # Step 1: Get the original image properties
    current_origin = image.GetOrigin()
    current_direction = image.GetDirection()
    current_size = image.GetSize()
    current_spacing = image.GetSpacing()
    
    # Step 2: Resize the image (without padding) to the target size and spacing
    resampler = sitk.ResampleImageFilter()
    resampled_size = [
        int(np.round(current_size[0] * (current_spacing[0] / new_spacing[0]))),
        int(np.round(current_size[1] * (current_spacing[1] / new_spacing[1]))),
        int(np.round(current_size[2] * (current_spacing[2] / new_spacing[2])))]
    resampler.SetSize(resampled_size)  # Set target size in pixels
    resampler.SetOutputSpacing(new_spacing)  # Set target spacing in physical units
    resampler.SetOutputOrigin(current_origin)
    resampler.SetOutputDirection(current_direction)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resized_image = resampler.Execute(blurred_image)
    resized_image = sitk.Cast(resized_image, sitk.sitkInt16) 
    sitk.WriteImage(resized_image,"./data/new_resampled.mhd")
    
    # Step 3: If the resized image is smaller than the output size, pad it to center
    resized_size = resized_image.GetSize()

    target_size = [192,256,256]

    resized_spacing = resized_image.GetSpacing()

    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]

    for i in range(3):
        diff = target_size[i] - resized_size[i]
        if diff > 0:
            pad_lower[i] = diff // 2
            pad_upper[i] = diff - pad_lower[i]

    padded_image = sitk.ConstantPad(resized_image, pad_lower, pad_upper, -1000)

    # Update origin after padding
    new_origin = list(resized_image.GetOrigin())
    for i in range(3):
        new_origin[i] -= pad_lower[i] * resized_spacing[i]

    padded_image.SetOrigin(new_origin)

    padded_size = padded_image.GetSize()
    # Step 4 : Compute the cropping if required on each side (left/right, top/bottom)
    cropping_x = min(0, target_size[0] - padded_size[0])
    cropping_y = min(0, target_size[1] - padded_size[1])
    cropping_z = min(0, target_size[2] - padded_size[2])

    if sum([cropping_x, cropping_y, cropping_z]) != 0:
        crop_left = abs(int(cropping_x // 2))
        crop_right = int(abs(cropping_x) - crop_left)
        crop_top = abs(int(cropping_y // 2))
        crop_bottom = int(abs(cropping_y) - crop_top)
        crop_toppp = abs(int(cropping_z // 2))
        crop_bottomm = int(abs(cropping_z) - crop_toppp)

        cropped_image = padded_image[
            crop_left:padded_size[0]-crop_right,
            crop_top:padded_size[1]-crop_bottom,
            crop_toppp:padded_size[2]-crop_bottomm
        ]

        # Update origin after cropping
        cropped_origin = list(padded_image.GetOrigin())
        spacing = padded_image.GetSpacing()

        cropped_origin[0] += crop_left * spacing[0]
        cropped_origin[1] += crop_top * spacing[1]
        cropped_origin[2] += crop_toppp * spacing[2]

        cropped_image.SetOrigin(cropped_origin)

        final_image = cropped_image
    else:
        final_image = padded_image
    
    final_image = sitk.SmoothingRecursiveGaussian(final_image, sigma=1.5)

    resized_image = final_image
    resized_size =  resized_image.GetSize()
    resized_direction = resized_image.GetDirection()
    resized_origin = resized_image.GetOrigin()
    resized_spacing = resized_image.GetSpacing()
    resized_image = sitk.SmoothingRecursiveGaussian(resized_image, sigma=1.5) ### Added

    sitk.WriteImage(resized_image, "./data/new_resized.mhd")
    log.info(f"Spacing resized : {resized_image.GetSpacing()} and shape {resized_image.GetSize()}")


    # Step 5 : Image to tensor
    image_array = sitk.GetArrayFromImage(resized_image)
    image_array_corrected = np.where(image_array < -1024, -1024, image_array)
    image_array_corrected = image_array_corrected.astype(np.int16)

    image_tensor = torch.tensor(image_array_corrected).float()
    image_tensor = image_tensor.unsqueeze(axis=0).unsqueeze(axis=0)
    image_tensor = image_tensor.to(device=params.device)
    params.input_size = image_array_corrected.shape
    del image_array_corrected
    del image_array

    # Step 6 : Prediction with the model(s) and post processing
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
        checkpoint = torch.load(params.pre_trained_weights_path_3D, map_location=params.device, weights_only=False)
        my_dic_keys = list(checkpoint["model_state_dict"].keys())
        for key in my_dic_keys:
            if 'module' in key:
                checkpoint["model_state_dict"][key[7:]]= checkpoint["model_state_dict"][key]
                del checkpoint["model_state_dict"][key]
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(params.device)
        model.eval()
        start = time()
        prediction = model(image_tensor)
        inference_time = time() - start
        prediction = torch.squeeze(prediction)  # remove batch dimension
        prediction = torch.argmax(prediction, axis=0) # retrieve mask 

    # Step 7 : Post processing to remove small non connected objects
    prediction = prediction.detach().cpu().numpy().astype(bool)
    label_img, nums = label(prediction, return_num=True, connectivity=2)
    props = regionprops_table(label_img, properties=("label","area"))
    min_areas = props['area'][props['area'] < int((params.input_size[2] / 15)**3)]
    if len(min_areas) > 0:
        min_comp_size = max(min_areas) + 1
        log.info(f"Elements kept after post-processing : {props['area'][props['area'] > min_comp_size]} in voxels")
        prediction = remove_small_objects(prediction, max_size=min_comp_size)
    else:
        pass

    # Step 8 : Saving of the prediction as a sitk image with the same metadata as the original image
    prediction_sitk = sitk.GetImageFromArray(prediction.astype(np.uint8))
    log.info(f"Shape pred_sitk : {prediction_sitk.GetSize()} and spacing {prediction_sitk.GetSpacing()}")

    # Copy spatial metadata from padded image
    prediction_sitk.CopyInformation(final_image)

    # Resample the prediction back to the original image space using nearest neighbor interpolation
    prediction_image = sitk.Resample(
        prediction_sitk,
        image,  # reference image
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0, # default pixel value for areas outside the original image
        sitk.sitkUInt8
    )
    post_traitement_time = time() - start_zero
    
    log.info(f"Spacing pred_orig : {prediction_image.GetSpacing()} and shape {prediction_image.GetSize()}")
    prediction_image.SetDirection(original_direction)
    prediction_image.SetOrigin(original_origin)
    prediction_image.SetSpacing(original_spacing)
    sitk.WriteImage(prediction_image, os.path.join("./results_showcase", params.output_img_path))

    # log.info(f'cuda memory : {torch.cuda.memory_allocated()  /1024/1024}')
    # log.info(mem_usage())
    log.info(f'Inference time {inference_time} and total time {post_traitement_time}')


if __name__ == '__main__':

    # Define parameters
    parser = argparse.ArgumentParser(description="Deep Learning based Lung Segmentation algorithm")
    parser.add_argument('-i', "--input", required=True, type=str, help="Path to input image")
    parser.add_argument('-o', "--output", required=True, type=str, help="Path to output image")
    parser.add_argument('-w',"--weights", required=True, type=str, help="Path to model weights")
    parser.add_argument('-d', "--device", required=False, default='cpu', type=str, help="device on which to run the code, by default cpu but can be gpu")
    parser.add_argument('-m', "--method", required=False, default='3D', type=str, help="Method to pick between '3D' and 'multi-2D', (default 3D)")
    parser.add_argument('-s', "--spacing", required=False, default=2.0, type=float, help="Spacing for the input image (default [1.0, 1.0, 1.0])")
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
    params.spacing = args.spacing

    main(params, log, logging_level=logging.INFO)
