import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F


def dgsc_loss(outputs, labels):

    logits = outputs["logits"]
    coarse_logits = outputs["coarse_logits"]
    foreground_sdf = outputs["foreground_sdf"]

    decoder_size = coarse_logits.shape[-2:]
    low_labels = F.interpolate(
        labels[:, None].float(), size=decoder_size, mode="nearest"
    ).squeeze(1).long()
    sdf_target = _signed_distance_targets(
        labels, logits.shape[1], decoder_size
    ).to(device=foreground_sdf.device)

    segmentation = _segmentation_loss(logits, labels)
    coarse = _segmentation_loss(coarse_logits, low_labels)
    sdf = _signed_distance_loss(foreground_sdf, sdf_target)
    boundary = _boundary_reclassification_loss(
        outputs["refined_logits"], low_labels, sdf_target
    )
    auxiliary = 0.2 * coarse + 0.1 * sdf + 0.1 * boundary
    return segmentation + auxiliary


def _segmentation_loss(logits, labels):
    ce = F.cross_entropy(logits, labels)
    probabilities = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    smooth = 1e-5
    dice_loss = 0.0
    for class_index in range(num_classes):
        score = probabilities[:, class_index]
        target = (labels == class_index).float()
        intersection = (score * target).sum()
        denominator = score.square().sum() + target.square().sum()
        dice_loss += 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return 0.4 * ce + 0.6 * (dice_loss / num_classes)


def _signed_distance_targets(labels, num_classes, decoder_size):
    max_distance = 16.0
    batch_targets = []
    for label in labels.detach().cpu().numpy():
        class_targets = []
        for class_index in range(1, num_classes):
            mask = label == class_index
            if not mask.any():
                signed = np.full(label.shape, max_distance, dtype=np.float32)
            elif mask.all():
                signed = np.full(label.shape, -max_distance, dtype=np.float32)
            else:
                signed = distance_transform_edt(~mask) - distance_transform_edt(mask)
                signed = np.sign(signed) * np.maximum(np.abs(signed) - 1.0, 0.0)
                signed = signed.astype(np.float32, copy=False)
            class_targets.append(np.clip(signed / max_distance, -1.0, 1.0))
        batch_targets.append(np.stack(class_targets))

    targets = torch.from_numpy(np.stack(batch_targets))
    if targets.shape[-2:] != decoder_size:
        targets = F.interpolate(targets, size=decoder_size, mode="bilinear", align_corners=False)
    return targets


def _signed_distance_loss(prediction, target):
    per_class = []
    for class_index in range(target.shape[1]):
        pred_class = prediction[:, class_index]
        target_class = target[:, class_index]
        weight = 1.0 + 4.0 * torch.exp(-target_class.abs() / 0.25)
        error = F.smooth_l1_loss(pred_class, target_class, reduction="none")
        regression = (weight * error).sum() / weight.sum().clamp_min(1.0)

        pred_dx = pred_class[:, :, 1:] - pred_class[:, :, :-1]
        target_dx = target_class[:, :, 1:] - target_class[:, :, :-1]
        pred_dy = pred_class[:, 1:, :] - pred_class[:, :-1, :]
        target_dy = target_class[:, 1:, :] - target_class[:, :-1, :]
        gradient = 0.5 * (
            F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)
        )

        present_fraction = (target_class < 0).flatten(1).any(dim=1).float().mean()
        class_weight = 0.1 + 0.9 * present_fraction
        per_class.append(class_weight * (regression + 0.5 * gradient))
    return torch.stack(per_class).mean()


def _boundary_reclassification_loss(refined_logits, low_labels, sdf_target):
    boundary = sdf_target.abs().amin(dim=1) <= 0.25
    pixel_ce = F.cross_entropy(refined_logits, low_labels, reduction="none")
    per_class = []
    for class_index in range(refined_logits.shape[1]):
        class_mask = boundary & (low_labels == class_index)
        if class_mask.any():
            per_class.append(pixel_ce[class_mask].mean())
    if not per_class:
        return refined_logits.sum() * 0.0
    return torch.stack(per_class).mean()
