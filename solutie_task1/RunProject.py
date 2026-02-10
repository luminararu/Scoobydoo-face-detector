from FacialDetector import *
import pdb
import os
import numpy as np
from Visualize import *
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

def load_image(f):
    return cv.imread(f, cv.IMREAD_COLOR)


params: Parameters = Parameters()
params.window_width = 64
params.window_height = 84
params.step = 8
params.threshold = 0.7
params.has_annotations = False

facial_detector: FacialDetector = FacialDetector(params)

# --- Pozitive ---
positive_files = glob.glob(os.path.join(params.dir_pos_examples, '*.jpg'))

# Citim imaginile în paralel
with ThreadPoolExecutor() as executor:
    positive_patches = list(executor.map(load_image, positive_files))

positive_patches = np.array(positive_patches)

# Flip dacă e activat
if params.use_flip_images:
    flipped_patches = np.array([cv.flip(patch, 1) for patch in positive_patches])
    positive_patches = np.concatenate((positive_patches, flipped_patches), axis=0)
    print(f'Am adaugat {len(flipped_patches)} patch-uri oglindite')

# === AUGMENTARE suplimentară pentru patch-uri pozitive ===
augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

augmented_patches = []
for patch in positive_patches:
    # transformările torchvision lucrează cu PIL, deci ToPILImage
    augmented_patch = augment(patch)
    # Convertim înapoi la numpy (float32) și valori 0-255 pentru compatibilitate
    augmented_patch = (augmented_patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    augmented_patches.append(augmented_patch)

# Adaugăm augmentările la setul pozitiv
positive_patches = np.concatenate((positive_patches, np.array(augmented_patches)), axis=0)
print(f'Am adaugat {len(augmented_patches)} patch-uri augmentate')

print('Am incarcat patch-urile pozitive')

# --- Negative ---
negative_files = glob.glob(os.path.join(params.dir_neg_examples, '*.jpg'))

with ThreadPoolExecutor() as executor:
    negative_patches = list(executor.map(load_image, negative_files))

negative_patches = np.array(negative_patches)
print('Am incarcat patch-urile negative')

# --- Concatenare finale ---
training_examples = np.concatenate((positive_patches, negative_patches), axis=0)
train_labels = np.concatenate((np.ones(len(positive_patches)), np.zeros(len(negative_patches))))

print("====Incepe Antrenarea====")
facial_detector.train_classifier(training_examples, train_labels)

print("====Incepe Detectarea====")
detections, scores, file_names = facial_detector.run()

print("===Incepe Evaluare===")

if params.has_annotations:
    #facial_detector.eval_detections(detections, scores, file_names, False)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)

