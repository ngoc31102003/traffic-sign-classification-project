# data_preprocessor.py
import pandas as pd
import os


class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = ['Cam', 'Chidan', 'Hieulenh', 'Nguyhiem', 'Phu']

    def create_annotations(self):
        """Tạo annotation files đơn giản"""
        for split in ['train', 'valid', 'test']:
            annotations = []
            split_path = os.path.join(self.dataset_path, split)

            for class_idx, class_name in enumerate(self.classes):
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            annotations.append({
                                'image_path': os.path.join(split, class_name, img_file),
                                'label': class_idx
                            })

            # Save to CSV
            df = pd.DataFrame(annotations)
            df.to_csv(f'data/{split}.csv', index=False)
            print(f"Created {split}.csv: {len(df)} images")


# Usage
preprocessor = DataPreprocessor('dataset')
preprocessor.create_annotations()