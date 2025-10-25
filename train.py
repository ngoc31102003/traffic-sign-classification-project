# train.py
# To run tensorboard: tensorboard --logdir runs --port 6006
# tensorboard --logdir C:\Users\admin\PycharmProjects\Hoc_DL_CV\traffic_sign_project\runs\traffic_sign_experiment --port 6006
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time

from data_loader import TrafficDataset, get_transforms
from model import TrafficModel


def train_model(resume=False):
    config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 30,
        'dataset_path': 'dataset'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter('runs/traffic_sign_experiment')

    # Data
    train_transform, val_transform = get_transforms()
    train_dataset = TrafficDataset('data/train.csv', config['dataset_path'], transform=train_transform, augment=True)
    val_dataset = TrafficDataset('data/valid.csv', config['dataset_path'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Model
    model = TrafficModel(num_classes=5).to(device)

    # Print model summary to TensorBoard
    try:
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
    except Exception as e:
        print(f"Could not add model graph to TensorBoard: {e}")

    # Resume training
    start_epoch = 0
    best_acc = 0
    if resume and os.path.exists('models/last_model.pth'):
        checkpoint = torch.load('models/last_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])

    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()

        # ===== TRAINING =====
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Train]')

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config["epochs"]} [Val]')

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Update validation progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        # ===== LOG TO TENSORBOARD =====
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time/Epoch', epoch_time, epoch)

        # ===== ADD TEXT SUMMARIES TO TENSORBOARD =====
        epoch_summary_text = f"""
Epoch {epoch + 1}/{config['epochs']} Summary:
Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%
Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%
Best Val Acc: {best_acc:.2f}%
Time: {epoch_time:.2f}s
Learning Rate: {optimizer.param_groups[0]['lr']:.6f}
        """

        # Add text summary to TensorBoard
        writer.add_text('Epoch Summary', epoch_summary_text, epoch)

        # Add detailed metrics as text
        writer.add_text('Metrics/Train', f'Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%', epoch)
        writer.add_text('Metrics/Validation', f'Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%', epoch)

        # Add individual scalar values for easy reading
        writer.add_scalar('Metrics/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Metrics/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Metrics/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Val_Accuracy', val_acc, epoch)

        # ===== PRINT SUMMARY =====
        print(f"\nEpoch {epoch + 1}/{config['epochs']} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc: {best_acc:.2f}%")

        # ===== SAVE MODELS =====
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, 'models/last_model.pth')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': config
            }, 'models/best_model.pth')
            print(f"NEW BEST! Validation Accuracy: {val_acc:.2f}%")

            # Log best accuracy to TensorBoard
            writer.add_scalar('Best_Accuracy', best_acc, epoch)
            writer.add_text('Best_Model', f'Best model at epoch {epoch + 1} with validation accuracy: {val_acc:.2f}%',
                            epoch)

        scheduler.step()
        print("-" * 50)

    # ===== TRAINING COMPLETED =====
    # Add final summary to TensorBoard
    final_summary = f"""
Training Completed!
Final Best Validation Accuracy: {best_acc:.2f}%
Total Epochs: {config['epochs']}
Final Training Accuracy: {train_acc:.2f}%
Final Validation Accuracy: {val_acc:.2f}%
    """
    writer.add_text('Final Summary', final_summary)
    writer.add_scalar('Final/Best_Accuracy', best_acc)
    writer.add_scalar('Final/Final_Train_Accuracy', train_acc)
    writer.add_scalar('Final/Final_Val_Accuracy', val_acc)

    writer.close()

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"TensorBoard logs saved in 'runs/' directory")
    print(f"To view TensorBoard, run: tensorboard --logdir runs --port 6006")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()

    train_model(resume=args.resume)