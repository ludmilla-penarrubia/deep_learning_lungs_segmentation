import json
import os
import torch
import SimpleITK as sitk
import numpy as np
from munch import Munch
from tqdm import tqdm

from src.model.unet import UNet
from src.utils.compute_prediction import compute_prediction
from skimage.morphology import remove_small_objects, label
from skimage.measure import regionprops_table

def main(params):

    params.results_folder = os.path.join("./results_showcase", params.exp_tag)
    os.makedirs(params.results_folder, exist_ok=True)

    # Load data
    image = sitk.ReadImage(params.input_img_path)
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    image_array = sitk.GetArrayFromImage(image)
    image_tensor = torch.tensor(image_array).float()
    image_tensor = image_tensor.unsqueeze(axis=0).unsqueeze(axis=0)
    image_tensor = image_tensor.to(device=params.device)
    params.input_size = image_array.shape

    # Create empty volume to sum predictions
    pred_sum = torch.zeros((params.input_size)).to(params.device)

    # Load UNet models and predict volumes in each direction
    for slicing in tqdm(params.pre_trained_weights_path.keys(), total=3, desc="Direction prediction"):
        model = UNet(params.n_channels, params.n_classes)
        checkpoint = torch.load(params.pre_trained_weights_path[slicing], map_location=params.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(params.device)
        model.eval()
        pred_array = compute_prediction(model, slicing, image_tensor)
        pred_sum += pred_array

    # Majority vote
    prediction = torch.where(pred_sum >= 2, 1, 0)

    # Post processing to remove small non connected objects
    prediction = prediction.detach().cpu().numpy().astype(bool)
    label_img, nums = label(prediction, return_num=True, connectivity=2)
    props = regionprops_table(label_img, properties=("label","area"))
    min_areas = props['area'][props['area'] < int((params.input_size[2] / 15)**3)]
    if len(min_areas) > 0:
        min_comp_size = max(min_areas) + 1
        print(f"Elements kept after post-processing : {props['area'][props['area'] > min_comp_size]} in voxels")
        merge_array_ = remove_small_objects(prediction, min_comp_size, in_place=True)
    else:
        pass

    # Saving of the prediction
    prediction_image = sitk.GetImageFromArray(prediction.astype(np.int8))
    prediction_image.SetDirection(direction)
    prediction_image.SetOrigin(origin)
    prediction_image.SetSpacing(spacing)
    sitk.WriteImage(prediction_image, os.path.join(params.results_folder, "pred_demo.mhd"))


if __name__ == '__main__':

    # Define parameters
    params = Munch()
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    params.input_img_path = "./data/input_data/Expi-B.mhd"
    params.n_channels = 1
    params.n_classes = 2
    params.pre_trained_weights_path = {
        "axial": "./data/model_weights/train_multi_PK352_CHU320_axial_fold0.pt",
        "coronal" : "./data/model_weights/train_multi_PK352_CHU320_coronal_fold0.pt",
        "sagittal" : "./data/model_weights/train_multi_PK352_CHU320_sagittal_fold0.pt",
    }

    # Extract the experiment tag and create the associated folder
    params.exp_tag = "trained_model_on_your_data"

    main(params)
