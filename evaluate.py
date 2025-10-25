# evaluate.py
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from data_loader import TrafficDataset, get_transforms
from model import TrafficModel


def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model - SỬA LỖI Ở ĐÂY
    model = TrafficModel(num_classes=5)

    try:
        # Thử load với weights_only=True trước
        checkpoint = torch.load('models/best_model.pth', map_location=device, weights_only=False)

        # Kiểm tra cấu trúc checkpoint
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            print("✅ Loaded model from 'model_state'")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded model from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("✅ Loaded model from 'state_dict'")
        else:
            # Nếu không có key nào phù hợp, thử load trực tiếp
            model.load_state_dict(checkpoint)
            print("✅ Loaded model directly from checkpoint")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Test data
    _, test_transform = get_transforms()
    test_dataset = TrafficDataset('data/test.csv', 'dataset', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # Evaluation
    all_preds = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Results
    classes = ['Cam', 'Chidan', 'Hieulenh', 'Nguyhiem', 'Phu']

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # Accuracy
    accuracy = 100. * sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_preds)
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(classes):
        class_mask = np.array(all_labels) == i
        if sum(class_mask) > 0:
            class_acc = 100. * sum(np.array(all_preds)[class_mask] == i) / sum(class_mask)
            print(f"  {class_name}: {class_acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Traffic Sign Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Additional metrics
    print("\n" + "=" * 60)
    print("ADDITIONAL METRICS")
    print("=" * 60)

    # Precision, Recall, F1 for each class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)

    for i, class_name in enumerate(classes):
        print(
            f"{class_name:10} - Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1[i]:.3f}, Support: {support[i]}")

    # Overall metrics
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support)
    print(f"\nMacro F1-score: {macro_f1:.3f}")
    print(f"Weighted F1-score: {weighted_f1:.3f}")

    # Save detailed results
    results = {
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_names': classes
    }

    # Print some misclassified examples
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)

    misclassified = []
    for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
        if pred != true:
            misclassified.append({
                'index': i,
                'predicted': classes[pred],
                'true': classes[true],
                'confidence': max(all_probabilities[i])
            })

    if misclassified:
        print(f"Number of misclassified samples: {len(misclassified)}")
        print("\nTop 10 misclassified samples (by confidence):")
        misclassified.sort(key=lambda x: x['confidence'], reverse=True)
        for i, item in enumerate(misclassified[:10]):
            print(
                f"  {i + 1}. True: {item['true']:10} -> Pred: {item['predicted']:10} (Conf: {item['confidence']:.3f})")
    else:
        print("No misclassified samples! Perfect accuracy!")


if __name__ == "__main__":
    evaluate_model()