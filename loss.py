import torch

import config
from utils import intersection_over_union


def yolo_loss(predicts, labels, scope='loss_layer'):
    """calculate loss function
    Args:
      predicts: 4-D tensor [batch_size, 8, 10, 5*nbox+n_class]
      labels: 4-D tensor [batch_size, 8, 10, 5+n_class]
    Return:
      loss: scalar
    """
    predicts = predicts.reshape(-1, config.out_W, config.out_H,
                                config.nclass + config.box_per_cell * 5)
    start_b1 = config.nclass + 1  # 4
    end_b1 = config.nclass + config.box_per_cell - 1  # 7
    start_b2 = end_b1 + 1  # 8
    end_b2 = start_b2 + 4  # 12
    start_label = start_b1
    end_lable = end_b1
    iou_b1 = intersection_over_union(
        predicts[..., start_b1:end_b1], labels[..., start_label:end_lable])
    iou_b2 = intersection_over_union(
        predicts[..., start_b2: end_b2], labels[..., start_label:end_lable])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

    iou_maxes, bestbox = torch.max(ious, dim=0)

    exists_box = labels[..., (start_label - 1)].unsqueeze(3)

    # ======================== #
    #   FOR BOX COORDINATES    #
    # ======================== #

    # Set boxes with no object in them to 0. We only take out one of the two
    # predictions, which is the one with highest Iou calculated previously.
    box_predictions = exists_box * (
        (
            bestbox * predicts[..., start_b2:end_b2]
            + (1 - bestbox) * predicts[..., start_b1:end_b1]
        )
    )

    box_targets = exists_box * labels[..., start_label:end_lable]

    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
        torch.abs(box_predictions[..., 2:4] + 1e-6)
    )
    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

    box_loss = torch.nn.MSELoss(
        torch.flatten(box_predictions, end_dim=-2),
        torch.flatten(box_targets, end_dim=-2),
    )

    # ==================== #
    #   FOR OBJECT LOSS    #
    # ==================== #

    # pred_box is the confidence score for the bbox with highest IoU
    pred_box = (
        bestbox * predicts[..., start_b2:end_b2] +
        (1 - bestbox) * predicts[..., (start_b1 - 1):start_b1]
    )

    object_loss = torch.nn.MSELoss(
        torch.flatten(exists_box * pred_box),
        torch.flatten(
            exists_box * labels[..., (start_label - 1): start_label]),
    )

    # ======================= #
    #   FOR NO OBJECT LOSS    #
    # ======================= #

    no_object_loss = torch.nn.MSELoss(
        torch.flatten((1 - exists_box) *
                      predicts[..., (start_b1 - 1): start_b1], start_dim=1),
        torch.flatten((1 - exists_box) *
                      labels[..., (start_label - 1): start_label], start_dim=1),
    )

    no_object_loss += torch.nn.MSELoss(
        torch.flatten((1 - exists_box) *
                      predicts[..., (start_b2 - 1):start_b2], start_dim=1),
        torch.flatten((1 - exists_box) *
                      labels[..., (start_label - 1): start_label], start_dim=1)
    )

    # ================== #
    #   FOR CLASS LOSS   #
    # ================== #

    class_loss = torch.nn.MSELoss(
        torch.flatten(
            exists_box * predicts[..., :(start_b1 - 1)], end_dim=-2,),
        torch.flatten(
            exists_box * labels[..., :(start_label - 1)], end_dim=-2,),
    )

    loss = (
        config.noobject_scale * box_loss  # first two rows in paper
        + object_loss  # third row in paper
        + config.box_scale * no_object_loss  # forth row
        + class_loss  # fifth row
    )

    return loss
