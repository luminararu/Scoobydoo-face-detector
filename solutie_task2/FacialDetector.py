from Parameters import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import os
import torch_directml
from torchvision import transforms

import torch.nn as nn



class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        # Transformări agresive pentru fețele personajelor (1-4)
        self.face_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((64, 84)),
            transforms.ToTensor(),  # aici se face /255
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # Transformări moderate pentru background / unknown (5)
        self.unknown_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(64, 84), scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx].item()

        if label in [0, 1, 2, 3, 4]:
            img = self.face_transform(img)
        else:
            img = self.unknown_transform(img)

        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25)
        )

        # Calculăm dimensiunea corectă după ultimele MaxPool (presupunând input 64x84)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 8 * 10, num_classes)  # 5 clase
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # nu aplicăm softmax, CrossEntropyLoss se ocupă


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        if torch_directml.is_available():
            self.device = torch_directml.device()
            print(f"✓ Folosesc GPU cu DirectML")
        else:
            self.device = torch.device('cpu')
            print("⚠ Folosesc CPU")

    def train_classifier(self, training_examples, train_labels):
        model_file = os.path.join(self.params.dir_save_files, 'cnn_face_classifier.pth')

        if os.path.exists(model_file):
            self.best_model = SimpleCNN(num_classes=6).to(self.device)
            self.best_model.load_state_dict(
                torch.load(model_file, map_location=self.device, weights_only=False)
            )
            self.best_model.eval()
            return

        # === Pregătire date ===
        X = torch.tensor(training_examples, dtype=torch.uint8) #/ 255.0
        X = X.permute(0, 3, 1, 2)
        y = torch.tensor(train_labels, dtype=torch.long)

        dataset = AugmentedDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # === Model ===
        model = SimpleCNN(num_classes=6).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # === Antrenare ===
        epochs = 20
        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0
            loss_epoch = 0.0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total
            print(f'Epoch {epoch + 1}/{epochs} | Loss: {loss_epoch:.4f} | Accuracy: {acc:.4f}')

        # === Salvare model ===
        torch.save(model.state_dict(), model_file)
        self.best_model = model
        self.best_model.eval()
        print(f"✓ Model salvat în {model_file}")

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        # Inițializare detecții pentru fiecare personaj
        # Fred (label=1)
        detections_fred = None
        scores_fred = np.array([])
        file_names_fred = np.array([])

        # Velma (label=2)
        detections_velma = None
        scores_velma = np.array([])
        file_names_velma = np.array([])

        # Shaggy (label=3)
        detections_shaggy = None
        scores_shaggy = np.array([])
        file_names_shaggy = np.array([])

        # Daphne (label=4)
        detections_daphne = None
        scores_daphne = np.array([])
        file_names_daphne = np.array([])

        window_h = 84
        window_w = 64
        step = 8

        self.best_model.eval()

        scales = [1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]

        for i, test_file in enumerate(test_files):
            start_time = timeit.default_timer()
            print(f'Procesam imaginea {i + 1}/{len(test_files)}')

            img = cv.imread(test_file, cv.IMREAD_COLOR)

            # Detecții separate pentru fiecare personaj în imagine curentă
            image_detections_fred = []
            image_scores_fred = []

            image_detections_velma = []
            image_scores_velma = []

            image_detections_shaggy = []
            image_scores_shaggy = []

            image_detections_daphne = []
            image_scores_daphne = []

            for scale in scales:
                scaled_img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

                if scaled_img.shape[0] < window_h or scaled_img.shape[1] < window_w:
                    continue

                for y in range(0, scaled_img.shape[0] - window_h, step):
                    for x in range(0, scaled_img.shape[1] - window_w, step):

                        patch = scaled_img[y:y + window_h, x:x + window_w]

                        patch = patch.astype(np.float32) / 255.0
                        patch = (patch - 0.5) / 0.5
                        patch = np.transpose(patch, (2, 0, 1))

                        # adaugă batch dimension și mută pe device
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            output = self.best_model(patch_tensor)  # ✓ Șters .item()

                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        score = probabilities[0, predicted_class].item()

                        if score > self.params.threshold:
                            x_min = int(x / scale)
                            y_min = int(y / scale)
                            x_max = int((x + window_w) / scale)
                            y_max = int((y + window_h) / scale)

                            detection = [x_min, y_min, x_max, y_max]

                            # Separă detecțiile pe personaje
                            if predicted_class == 1:  # Fred
                                image_detections_fred.append(detection)
                                image_scores_fred.append(score)
                            elif predicted_class == 2:  # Velma
                                image_detections_velma.append(detection)
                                image_scores_velma.append(score)
                            elif predicted_class == 3:  # Shaggy
                                image_detections_shaggy.append(detection)
                                image_scores_shaggy.append(score)
                            elif predicted_class == 4:  # Daphne
                                image_detections_daphne.append(detection)
                                image_scores_daphne.append(score)

            short_name = ntpath.basename(test_file)

            # NMS și adăugare pentru Fred
            if len(image_scores_fred) > 0:
                image_detections_fred, image_scores_fred = self.non_maximal_suppression(
                    np.array(image_detections_fred),
                    np.array(image_scores_fred),
                    img.shape
                )
                if detections_fred is None:
                    detections_fred = image_detections_fred
                else:
                    detections_fred = np.concatenate((detections_fred, image_detections_fred))
                scores_fred = np.append(scores_fred, image_scores_fred)
                file_names_fred = np.append(file_names_fred, [short_name] * len(image_scores_fred))

            # NMS și adăugare pentru Velma
            if len(image_scores_velma) > 0:
                image_detections_velma, image_scores_velma = self.non_maximal_suppression(
                    np.array(image_detections_velma),
                    np.array(image_scores_velma),
                    img.shape
                )
                if detections_velma is None:
                    detections_velma = image_detections_velma
                else:
                    detections_velma = np.concatenate((detections_velma, image_detections_velma))
                scores_velma = np.append(scores_velma, image_scores_velma)
                file_names_velma = np.append(file_names_velma, [short_name] * len(image_scores_velma))

            # NMS și adăugare pentru Shaggy
            if len(image_scores_shaggy) > 0:
                image_detections_shaggy, image_scores_shaggy = self.non_maximal_suppression(
                    np.array(image_detections_shaggy),
                    np.array(image_scores_shaggy),
                    img.shape
                )
                if detections_shaggy is None:
                    detections_shaggy = image_detections_shaggy
                else:
                    detections_shaggy = np.concatenate((detections_shaggy, image_detections_shaggy))
                scores_shaggy = np.append(scores_shaggy, image_scores_shaggy)
                file_names_shaggy = np.append(file_names_shaggy, [short_name] * len(image_scores_shaggy))

            # NMS și adăugare pentru Daphne
            if len(image_scores_daphne) > 0:
                image_detections_daphne, image_scores_daphne = self.non_maximal_suppression(
                    np.array(image_detections_daphne),
                    np.array(image_scores_daphne),
                    img.shape
                )
                if detections_daphne is None:
                    detections_daphne = image_detections_daphne
                else:
                    detections_daphne = np.concatenate((detections_daphne, image_detections_daphne))
                scores_daphne = np.append(scores_daphne, image_scores_daphne)
                file_names_daphne = np.append(file_names_daphne, [short_name] * len(image_scores_daphne))

            end_time = timeit.default_timer()
            print(f'Timp procesare: {end_time - start_time:.2f} sec')

        # Returnare rezultate separate pentru fiecare personaj
        return (
            (detections_fred, scores_fred, file_names_fred),
            (detections_velma, scores_velma, file_names_velma),
            (detections_shaggy, scores_shaggy, file_names_shaggy),
            (detections_daphne, scores_daphne, file_names_daphne)
        )

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names, mining):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        if mining == True:
            ground_truth_detections = np.array(ground_truth_file[:, 1:-1], dtype=int)
        else:
            ground_truth_detections = np.array(ground_truth_file[:, 1:], dtype = int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)

        # === ACCURATEȚE PE VALIDARE (DETECTION ACCURACY) ===
        val_accuracy = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
        print(f"Validation detection accuracy: {val_accuracy:.4f}")

        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

    def eval_detections_character(self, detections, scores, file_names, ground_truth_path, character):
        ground_truth_path = os.path.join(self.params.base_dir_valid, ground_truth_path)
        ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)

        # === ACCURATEȚE PE VALIDARE (DETECTION ACCURACY) ===
        val_accuracy = np.sum(true_positive) / (np.sum(true_positive) + np.sum(false_positive))
        print(f"Validation detection accuracy: {val_accuracy:.4f}")

        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(character + ' faces: average precision %.3f' % average_precision)
        plt.savefig('precizie_medie_' + character + '.png')
        plt.show()

    def evaluate_results_task2(self, solution_path, ground_truth_path, character, verbose=0):

        # incarca detectiile + scorurile + numele de imagini
        detections = np.load(solution_path + "detections_all_faces_" + character + ".npy", allow_pickle=True, fix_imports=True,
                             encoding='latin1')
        print(detections.shape)

        scores = np.load(solution_path + "scores_all_faces_" + character + ".npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
        print(scores.shape)

        file_names = np.load(solution_path + "file_names_all_faces_" + character + ".npy", allow_pickle=True, fix_imports=True,
                             encoding='latin1')
        print(file_names.shape)

        self.eval_detections_character(detections, scores, file_names, ground_truth_path, character)

    def save_false_positive_patches(self, false_positive_detections, false_positive_file_names, false_positive_scores, images):
        """
        Extrage HOG features pentru false positives și le salvează într-un .npy
        """
        negative_features = []

        return negative_features

    def detections_hard_mining(self, detections, scores, file_names, annotations, images):

        path = os.path.join(self.params.base_dir, annotations)
        ground_truth_file = np.loadtxt(path, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:-1], dtype=int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    #false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        false_positive_mask = (false_positive == 1) & (scores > 2)
        false_positive_detections = detections[false_positive_mask]
        false_positive_file_names = file_names[false_positive_mask]
        false_positive_scores = scores[false_positive_mask]


        saved_count = self.save_false_positive_patches(false_positive_detections, false_positive_file_names, false_positive_scores, images)

        print(f"Total patch-uri false positive salvate: {saved_count}")
        return false_positive_detections, false_positive_scores, false_positive_file_names

