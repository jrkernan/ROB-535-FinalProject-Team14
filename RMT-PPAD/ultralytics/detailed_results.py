import torch
import yaml
import numpy as np
from ultralytics import MTDETR
from ultralytics.models.mtdetr.val import MTDETRValidator

MODEL_PATH = "/home/jrkernan/project/RMT-PPAD/ultralytics/best.pt"
DATA_YAML = "/home/jrkernan/project/RMT-PPAD/ultralytics/cfg/datasets/BDD_full.yaml"
DEVICE = 0
IMG_SIZE = 640
BATCH = 1
MASK_THRESH = [0.45, 0.9]   # use same thresholds as paper

def run_val_with_f1():
    
    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    number_task = {
        "detection": len(data_cfg["type_task"]["detection"]),
        "segmentation": len(data_cfg["type_task"]["segmentation"]),
    }
    # Build args exactly like Ultralytics test.py
    args = dict(
        model=MODEL_PATH,
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=[DEVICE],
        task="multi",
        split="val",
        mask_threshold=MASK_THRESH,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
        verbose=False,
        save_json=False,
        save_txt=False,
        single_cls=False,
        half=False,
        cache=False,
        workers=4,
        overlap_mask=False,
        mask_ratio=1,
    )

    # Load model
    model = MTDETR(MODEL_PATH)
    model.model = model.model.to(model.device)

    # Build validator (this constructs dataset + dataloader correctly)
    validator = MTDETRValidator(
        args=args,
        number_task=number_task
    )
    validator.model = model.model
    validator.device = model.device

    # RUN VALIDATION (this populates seg_metrics + seg_result)
    validator()
    
    validator.model = validator.model.to(validator.device)
    validator.model.eval()
    
    conf_mats = {
        task: np.zeros((2, 2), dtype=np.float64)
        for task in validator.seg_metrics.keys()
    }
    
    with torch.no_grad():
        for batch in validator.dataloader:
            imgs = batch["img"].to(validator.device).float() / 255.0

            preds = validator.model(imgs)
            _, pred_masks = validator.postprocess(preds)

            # pred_masks shape: [B, num_tasks, H, W]
            for task_idx, task_name in enumerate(conf_mats.keys()):
                pred = pred_masks[0, task_idx].cpu().numpy()
                gt = batch["merge_mask"][0, task_idx].cpu().numpy()

                # ensure binary
                pred = (pred > 0).astype(np.uint8)
                gt = (gt > 0).astype(np.uint8)
    
                tp = np.sum((pred == 1) & (gt == 1))
                fp = np.sum((pred == 1) & (gt == 0))
                fn = np.sum((pred == 0) & (gt == 1))
                tn = np.sum((pred == 0) & (gt == 0))

                conf_mats[task_name] += np.array([
                    [tn, fp],
                    [fn, tp]
                ])

    print("\n=== Segmentation Metrics (with F1) ===")

    for task_name, cm in conf_mats.items():
        tn, fp = cm[0]
        fn, tp = cm[1]

        pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)

        print(f"\nTask: {task_name}")
        print(f"  Pixel Acc : {pixel_acc:.4f}")
        print(f"  IoU       : {iou:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")


if __name__ == "__main__":
    run_val_with_f1()