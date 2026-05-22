import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)


def get_data_transforms():
    return {
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


def load_datasets(data_dir='./data', batch_size=64, num_workers=4):
    data_transforms = get_data_transforms()

    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    validation_dataset = datasets.ImageFolder(validation_dir, transform=data_transforms['validation'])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return train_dataset, validation_dataset, train_loader, validation_loader


def compute_class_weights(dataset, num_classes, device):
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[label] += 1

    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * max(1, c)) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    return class_weights


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True, ncols=120)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            known_mask = ~unknown_mask

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted[known_mask] == labels[known_mask]).sum().item()
            unknown_count += unknown_mask.sum().item()

    known_total = total - unknown_count
    accuracy = 100 * correct / known_total if known_total > 0 else 0.0
    validation_loss = running_loss / len(dataloader)

    return validation_loss, accuracy, unknown_count


def cleanup_old_models(best_model_filename, prefix, model_dir='.'):
    for f in os.listdir(model_dir):
        if f.startswith(prefix) and f.endswith('.pth') and f != best_model_filename:
            old_path = os.path.join(model_dir, f)
            os.remove(old_path)
            logger.info(f"Eski model silindi: {old_path}")


def print_epoch_summary(epoch, num_epochs, train_loss, train_acc, validation_loss, accuracy,
                        unknown_count, best_accuracy, current_lr, improved, epochs_without_improvement,
                        early_stopping_patience):
    status_icon = "🟢 YENİ BEST" if improved else f"🔴 İyileşme yok ({epochs_without_improvement}/{early_stopping_patience})"

    print(flush=True)
    print("═" * 90, flush=True)
    print(f"  📊 EPOCH {epoch + 1}/{num_epochs}  |  {status_icon}", flush=True)
    print("─" * 90, flush=True)
    print(f"  Train Loss:  {train_loss:.4f}    │  Train Acc:   {train_acc:.2f}%", flush=True)
    print(f"  Val Loss:    {validation_loss:.4f}    │  Val Acc:     {accuracy:.2f}%", flush=True)
    print(f"  Unknown:     {unknown_count:<10}   │  Best Acc:    {max(best_accuracy, accuracy):.2f}%", flush=True)
    print(f"  LR (backbone): {current_lr:.2e}", flush=True)
    print("═" * 90, flush=True)


def print_training_complete(arch_name, best_accuracy, best_loss, best_model_filename):
    print(flush=True)
    print("╔" + "═" * 88 + "╗", flush=True)
    print("║" + f"  {arch_name} EĞİTİM TAMAMLANDI".center(88) + "║", flush=True)
    print("╠" + "═" * 88 + "╣", flush=True)
    print(f"║  En iyi doğruluk:  {best_accuracy:.2f}%".ljust(89) + "║", flush=True)
    print(f"║  En düşük loss:    {best_loss:.4f}".ljust(89) + "║", flush=True)
    print(f"║  Model dosyası:    {best_model_filename}".ljust(89) + "║", flush=True)
    print("╚" + "═" * 88 + "╝", flush=True)


def run_training_loop(model, train_loader, validation_loader, criterion, optimizer, scheduler,
                      scaler, device, backbone_params, num_epochs, unknown_threshold,
                      early_stopping_patience, freeze_epochs, model_prefix, arch_name,
                      class_names, num_classes, save_extra=None):
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_filename = None
    epochs_without_improvement = 0

    logger.info(f"{arch_name} eğitim başlıyor: {num_epochs} epoch, early stopping patience={early_stopping_patience}")
    logger.info(f"Aşama 1: İlk {freeze_epochs} epoch — sadece classifier katmanı eğitiliyor (backbone donduruldu)")

    for param in backbone_params:
        param.requires_grad = False

    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            logger.info(f"Aşama 2: Epoch {epoch+1} — backbone çözüldü, tüm model fine-tune ediliyor")
            for param in backbone_params:
                param.requires_grad = True

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs
        )

        validation_loss, accuracy, unknown_count = validate(
            model, validation_loader, criterion, device, unknown_threshold
        )

        scheduler.step(epoch + validation_loss / (validation_loss + 1))

        current_lr = optimizer.param_groups[0]['lr']
        improved = accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_loss)

        print_epoch_summary(
            epoch, num_epochs, train_loss, train_acc, validation_loss, accuracy,
            unknown_count, best_accuracy, current_lr, improved,
            epochs_without_improvement + (0 if improved else 1), early_stopping_patience
        )

        if improved:
            best_accuracy = accuracy
            best_loss = validation_loss
            epochs_without_improvement = 0

            new_filename = f"{model_prefix}_epoch_{epoch + 1}_acc_{best_accuracy:.2f}.pth"
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'accuracy': best_accuracy,
                'loss': best_loss,
                'class_names': class_names,
                'num_classes': num_classes,
                'unknown_threshold': unknown_threshold,
            }
            if save_extra:
                save_dict.update(save_extra)
            torch.save(save_dict, new_filename)
            print(f"  💾 Model kaydedildi → {new_filename}", flush=True)

            if best_model_filename and best_model_filename != new_filename:
                cleanup_old_models(new_filename, model_prefix)
            best_model_filename = new_filename
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(flush=True)
                print("⛔ EARLY STOPPING!", flush=True)
                print(f"  {early_stopping_patience} epoch boyunca iyileşme yok.", flush=True)
                print(f"  En iyi doğruluk: {best_accuracy:.2f}%", flush=True)
                break

    print_training_complete(arch_name, best_accuracy, best_loss, best_model_filename)
