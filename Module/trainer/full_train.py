import torch
from tqdm import tqdm
from torch.nn.functional import threshold
from statistics import mean

from ..utils.metric import calculate_accuracy

def train_one_epoch(model,
                    data_loader,
                    loss_fn,
                    optimizer,
                    device='cuda',
                    ):

    model.train()

    train_epoch_acc_list = []
    train_epoch_loss_list = []
    for sample in tqdm(iter(data_loader)):
        train_image, train_gt, input_size, original_image_size, points, point_labels = sample
        train_image = train_image.to(device)
        train_gt = train_gt.to(device)
        points = points.to(device)
        point_labels = point_labels.to(device)

        image_embedding = model.image_encoder(train_image)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points, point_labels),
            boxes=None,
            masks=None,
        )

        pred_mask, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)

        pred_mask = model.postprocess_masks(pred_mask,
                                            (1024, 1024),
                                            (1024, 1024)).to(device)

        pred_mask = (threshold(torch.sigmoid(pred_mask), 0.5, 0))

        train_gt = train_gt.unsqueeze(1)
        train_gt = train_gt > 0.5
        train_gt = torch.as_tensor(train_gt > 0, dtype=torch.float32)

        loss = loss_fn(pred_mask, train_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss_list.append(loss.item())
        train_accuracy = calculate_accuracy(pred_mask, train_gt)
        train_epoch_acc_list.append(train_accuracy)
        
    mean_train_loss = mean(train_epoch_loss_list)
    mean_train_accuracy = mean(train_epoch_acc_list)

    return mean_train_loss, mean_train_accuracy