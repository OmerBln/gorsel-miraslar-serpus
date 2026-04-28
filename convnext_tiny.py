import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True, ncols=120)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, unknown_threshold):
    model.eval()
    correct = 0
    total = 0
    unknown_count = 0
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="[Validation]", leave=True, ncols=120)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probabilities = F.softmax(outputs.float(), dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            unknown_mask = max_probs < unknown_threshold
            predicted[unknown_mask] = -1

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            unknown_count += unknown_mask.sum().item()

    accuracy = 100 * correct / total
    validation_loss = running_loss / len(dataloader)

    return validation_loss, accuracy, unknown_count


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


def cleanup_old_models(best_model_filename, model_dir='.'):
    for f in os.listdir(model_dir):
        if f.startswith('best_model_convnext_tiny_') and f.endswith('.pth') and f != best_model_filename:
            old_path = os.path.join(model_dir, f)
            os.remove(old_path)
            logger.info(f"Eski model silindi: {old_path}")


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = './data'
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    validation_dataset = datasets.ImageFolder(validation_dir, transform=data_transforms['validation'])

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cihaz: {device}")

    num_classes = len(train_dataset.classes)
    model = build_model(num_classes, device)

    class_names = train_dataset.classes
    logger.info(f"Sınıflar ({num_classes}): {class_names}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Toplam parametre: {total_params:,} | Eğitilebilir: {trainable_params:,}")

    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1
        
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * max(1, c)) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    logger.info(f"Sınıf ağırlıkları: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = get_backbone_params(model)
    classifier_params = get_classifier_params(model)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': classifier_params, 'lr': 1e-3}
    ], weight_decay=0.01)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

    scaler = GradScaler()

    NUM_EPOCHS = 30
    UNKNOWN_THRESHOLD = 0.5
    EARLY_STOPPING_PATIENCE = 10
    FREEZE_EPOCHS = 3

    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_filename = None
    epochs_without_improvement = 0

    logger.info(f"ConvNeXt-Tiny eğitim başlıyor: {NUM_EPOCHS} epoch, early stopping patience={EARLY_STOPPING_PATIENCE}")

    logger.info(f"Aşama 1: İlk {FREEZE_EPOCHS} epoch — sadece classifier katmanı eğitiliyor (backbone donduruldu)")
    for param in backbone_params:
        param.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            logger.info(f"Aşama 2: Epoch {epoch+1} — backbone çözüldü, tüm model fine-tune ediliyor")
            for param in backbone_params:
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, NUM_EPOCHS
        )

        validation_loss, accuracy, unknown_count = validate(
            model, validation_loader, criterion, device, UNKNOWN_THRESHOLD
        )

        scheduler.step(epoch + validation_loss / (validation_loss + 1))

        current_lr = optimizer.param_groups[0]['lr']
        improved = accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_loss)
        status_icon = "🟢 YENİ BEST" if improved else f"🔴 İyileşme yok ({epochs_without_improvement + 1}/{EARLY_STOPPING_PATIENCE})"

        print(flush=True)
        print("═" * 90, flush=True)
        print(f"  📊 EPOCH {epoch + 1}/{NUM_EPOCHS}  |  {status_icon}", flush=True)
        print("─" * 90, flush=True)
        print(f"  Train Loss:  {train_loss:.4f}    │  Train Acc:   {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:    {validation_loss:.4f}    │  Val Acc:     {accuracy:.2f}%", flush=True)
        print(f"  Unknown:     {unknown_count:<10}   │  Best Acc:    {max(best_accuracy, accuracy):.2f}%", flush=True)
        print(f"  LR (backbone): {current_lr:.2e}", flush=True)
        print("═" * 90, flush=True)

        if improved:
            best_accuracy = accuracy
            best_loss = validation_loss
            epochs_without_improvement = 0

            new_filename = f"best_model_convnext_tiny_epoch_{epoch + 1}_acc_{best_accuracy:.2f}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'accuracy': best_accuracy,
                'loss': best_loss,
                'class_names': class_names,
                'num_classes': num_classes,
                'unknown_threshold': UNKNOWN_THRESHOLD,
                'architecture': 'convnext_tiny',
            }, new_filename)
            print(f"  💾 Model kaydedildi → {new_filename}", flush=True)

            if best_model_filename and best_model_filename != new_filename:
                cleanup_old_models(new_filename)
            best_model_filename = new_filename
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(flush=True)
                print("⛔ EARLY STOPPING!", flush=True)
                print(f"  {EARLY_STOPPING_PATIENCE} epoch boyunca iyileşme yok.", flush=True)
                print(f"  En iyi doğruluk: {best_accuracy:.2f}%", flush=True)
                break

    print(flush=True)
    print("╔" + "═" * 88 + "╗", flush=True)
    print("║" + "  CONVNeXT-TINY EĞİTİM TAMAMLANDI".center(88) + "║", flush=True)
    print("╠" + "═" * 88 + "╣", flush=True)
    print(f"║  En iyi doğruluk:  {best_accuracy:.2f}%".ljust(89) + "║", flush=True)
    print(f"║  En düşük loss:    {best_loss:.4f}".ljust(89) + "║", flush=True)
    print(f"║  Model dosyası:    {best_model_filename}".ljust(89) + "║", flush=True)
    print("╚" + "═" * 88 + "╝", flush=True)


if __name__ == '__main__':
    main()
