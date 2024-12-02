import torch
import numpy as np
from tqdm import tqdm

from ..sam import SamPredictor
from ..utils.metric import  calculate_accuracy

def val_one_epoch(model,
                  data_loader,
                  loss_fn,
                  device="cuda"):

    predictor_tuned = SamPredictor(model)

    val_loss = 0.0
    val_accuracy = 0.0
    num_val_examples = 0

    with torch.no_grad():
        for val_sample in tqdm(iter(data_loader)):
            val_img, val_gt = val_sample

            predictor_tuned.set_image(np.array(val_img[0]))

            pred_mask, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=None,
                multimask_output=False,
            )
            pred_mask = torch.as_tensor(pred_mask > 0, dtype=torch.float32)
            pred_mask = pred_mask.unsqueeze(0).to(device)

            val_gt = val_gt.unsqueeze(1).to(device)
            val_gt = val_gt > 0.5
            val_gt = torch.as_tensor(val_gt > 0, dtype=torch.float32)

            # Calculate validation loss
            val_loss += loss_fn(pred_mask, val_gt).item()
            # Calculate accuracy for validation data
            val_accuracy += calculate_accuracy(pred_mask, val_gt)
            num_val_examples += 1
    val_loss /= num_val_examples
    mean_val_accuracy = val_accuracy / num_val_examples

    return val_loss, mean_val_accuracy