import torch
import numpy as np
from ultralytics import YOLO

from ultralytics.yolo.utils import ops
from ultralytics.yolo.data.dataloader import create_dataloader

def calculate_AP(model_path, data_path, imgsz=640, conf_thres=0.001, iou_thres=0.5, target_class=None):
    # Load the model
    model = YOLO(model_path)

    # Prepare the dataset
    data = {
        'val': data_path
    }
    dataloader = create_dataloader(data['val'], imgsz, 1, 64, pad=0.5, rect=True, prefix='val: ')[0]

    # Set device
    device = model.device

    # Prepare metrics
    iouv = torch.tensor([0.5], device=device)
    stats = []

    # Iterate over the dataset
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True).float() / 255.0  # Convert to FP32
        nb, _, height, width = img.shape

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)

        # Apply NMS
        pred = ops.non_max_suppression(pred, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=True)

        for si, pred in enumerate(pred):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, iouv.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], iouv.numel(), dtype=torch.bool, device=device)

            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = ops.xywh2xyxy(labels[:, 1:5]) * torch.Tensor([width, height, width, height]).to(device)
                
                for cls in torch.unique(tcls_tensor):
                    ti = (tcls_tensor == cls).nonzero(as_tuple=False).view(-1)
                    pi = (pred[:, 5] == cls).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = ops.box_iou(pred[pi, :4], tbox[ti]).max(1)
                        detected_set = set()

                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ops.ap_per_class(*stats)
        if target_class is not None:
            i = (ap_class == target_class).nonzero(as_tuple=False).squeeze()
            ap_target = ap[i]
            print(f"AP for class {target_class}: {ap_target.mean():.4f}")
        else:
            ap50 = ap.mean(1)
            print(f"mAP@0.5: {ap50.mean():.4f}")
    else:
        print('No detections found.')

# Example usage:
calculate_AP('./runs/detect/train/weights/best.pt', '/voc.yaml', imgsz=640, target_class=0)