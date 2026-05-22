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
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    in_features = model.classifier[2].in_features  # 768

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)
    return model


def get_classifier_params(model):
    classifier_params = list(model.classifier.parameters())
    return classifier_params


def get_backbone_params(model):
    backbone_params = list(model.features.parameters())
    return backbone_params


def main():
    train_dataset, validation_dataset, train_loader, validation_loader = load_datasets()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cihaz: {device}")

    num_classes = len(train_dataset.classes)
    model = build_model(num_classes, device)

    class_names = train_dataset.classes
    logger.info(f"Sınıflar ({num_classes}): {class_names}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Toplam parametre: {total_params:,} | Eğitilebilir: {trainable_params:,}")

    class_weights = compute_class_weights(train_dataset, num_classes, device)
    logger.info(f"Sınıf ağırlıkları: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = get_backbone_params(model)
    classifier_params = get_classifier_params(model)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': classifier_params, 'lr': 1e-3}
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
        model_prefix='best_model_convnext_tiny',
        arch_name='CONVNeXT-TINY',
        class_names=class_names,
        num_classes=num_classes,
        save_extra={'architecture': 'convnext_tiny'},
    )


if __name__ == '__main__':
    main()
