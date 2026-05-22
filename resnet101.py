import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler
import logging

from train_utils import load_datasets, compute_class_weights, run_training_loop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def build_model(num_classes, device):
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)
    return model


def main():
    train_dataset, validation_dataset, train_loader, validation_loader = load_datasets()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cihaz: {device}")

    num_classes = len(train_dataset.classes)
    model = build_model(num_classes, device)

    class_names = train_dataset.classes
    logger.info(f"Sınıflar ({num_classes}): {class_names}")

    class_weights = compute_class_weights(train_dataset, num_classes, device)
    logger.info(f"Sınıf ağırlıkları: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    fc_params = list(model.fc.parameters())
    fc_param_ids = {id(p) for p in fc_params}
    backbone_params = [p for p in model.parameters() if id(p) not in fc_param_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': fc_params, 'lr': 1e-3}
    ], weight_decay=0.01)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    run_training_loop(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        backbone_params=backbone_params,
        num_epochs=30,
        unknown_threshold=0.5,
        early_stopping_patience=10,
        freeze_epochs=3,
        model_prefix='best_model_resnet101',
        arch_name='RESNET-101',
        class_names=class_names,
        num_classes=num_classes,
    )


if __name__ == '__main__':
    main()
