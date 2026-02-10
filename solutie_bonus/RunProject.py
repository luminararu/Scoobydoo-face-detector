import os
import shutil
import cv2
import glob
import yaml
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class YOLODataset:
    CLASSES = ['daphne', 'fred', 'shaggy', 'velma']
    SPLIT = 0.8

    def __init__(self, src='antrenare', dst='data_yolo'):
        self.src = src
        self.dst = dst

    def creare_foldere(self):
        if os.path.exists(self.dst):
            shutil.rmtree(self.dst)

        for t in ['train', 'val']:
            os.makedirs(f'{self.dst}/images/{t}', exist_ok=True)
            os.makedirs(f'{self.dst}/labels/{t}', exist_ok=True)

    def to_yolo(self, box, w, h):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / (2 * w)
        cy = (y1 + y2) / (2 * h)
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return cx, cy, bw, bh

    def annotations(self):
        data = []
        for cls in self.CLASSES:
            file = f'{self.src}/{cls}_annotations.txt'
            if not os.path.exists(file):
                continue
            for line in open(file):
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                img = f"{self.src}/{cls}/{parts[0]}"
                if not os.path.exists(img):
                    continue
                box = list(map(int, parts[1:5]))
                label = self.CLASSES.index(parts[5].lower())
                data.append((img, box, label))

        return data

    def export(self, records, img_dir, lbl_dir):
        grouped = defaultdict(list)
        for img, box, cls in records:
            grouped[img].append((box, cls))

        for i, (img_path, items) in enumerate(grouped.items()):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            cv2.imwrite(f'{img_dir}/{i}.jpg', img)
            with open(f'{lbl_dir}/{i}.txt', 'w') as f:
                for box, cls in items:
                    cx, cy, bw, bh = self.to_yolo(box, w, h)
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _yaml(self):
        cfg = {
            'path': os.path.abspath(self.dst),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.CLASSES),
            'names': self.CLASSES
        }
        with open(f'{self.dst}/config.yaml', 'w') as f:
            yaml.dump(cfg, f)

    def build(self):
        self.creare_foldere()
        data = self.annotations()

        np.random.shuffle(data)
        split = int(len(data) * self.SPLIT)

        self.export(data[:split], f'{self.dst}/images/train', f'{self.dst}/labels/train')
        self.export(data[split:], f'{self.dst}/images/val', f'{self.dst}/labels/val')
        self._yaml()
        print("Dataset creat!")

class CharacterDetector:
    LABELS = ['daphne', 'fred', 'shaggy', 'velma']

    def __init__(self, out='salveazaBonus'):
        self.model = None
        self.out = out
        os.makedirs(out, exist_ok=True)

    def train(self, data_yaml, epochs=20):
        model = YOLO('yolov8n.pt')
        model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16)
        self.model = model
        return model.trainer.best

    def load(self, weights):
        self.model = YOLO(weights)

    def predict(self, test_dir):
        images = glob.glob(f'{test_dir}/*.jpg')
        results = defaultdict(lambda: {'boxes': [], 'scores': [], 'files': []})
        all_boxes, all_scores, all_files = [], [], []

        for img in images:
            predictie = self.model.predict(img, conf=0.25, iou=0.45, verbose=False)[0]

            for box, score, cls in zip(predictie.boxes.xyxy.cpu().numpy(), predictie.boxes.conf.cpu().numpy(), predictie.boxes.cls.cpu().numpy().astype(int)):
                name = self.LABELS[cls]
                file = os.path.basename(img)

                results[name]['boxes'].append(box)
                results[name]['scores'].append(score)
                results[name]['files'].append(file)

                all_boxes.append(box)
                all_scores.append(score)
                all_files.append(file)

        for k, v in results.items():
            np.save(f'{self.out}/detections_{k}.npy', np.array(v['boxes']))
            np.save(f'{self.out}/scores_{k}.npy', np.array(v['scores']))
            np.save(f'{self.out}/file_names_{k}.npy', np.array(v['files']))

        np.save(f'{self.out}/detections_all_faces.npy', np.array(all_boxes))
        np.save(f'{self.out}/scores_all_faces.npy', np.array(all_scores))
        np.save(f'{self.out}/file_names_all_faces.npy', np.array(all_files))
        print("Submisie generata!")


'''
builder = YOLODataset()
builder.build()
'''
detector = CharacterDetector()
'''
best_weights = detector.train('data_yolo/config.yaml', epochs=15)
print("Best weights:", best_weights)
'''

detector.load("best.pt")
detector.predict("../validare/validare")
