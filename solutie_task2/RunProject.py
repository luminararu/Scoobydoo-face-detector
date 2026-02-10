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
params.has_annotations = True





facial_detector: FacialDetector = FacialDetector(params)




# --- Pozitive ---
def load_image(file_path):
    return cv.imread(file_path, cv.IMREAD_COLOR)

# --- Fred (label = 1) ---
positive_files_fred = glob.glob(os.path.join(params.dir_pos_examples_fred, '*.jpg'))
with ThreadPoolExecutor() as executor:
    positive_patches_fred = list(executor.map(load_image, positive_files_fred))
positive_patches_fred = np.array(positive_patches_fred)
labels_fred = np.ones(len(positive_patches_fred), dtype=np.int64) * 1  #
print(f'Am încărcat {len(positive_patches_fred)} patch-uri Fred')

# --- Velma (label = 2) ---
positive_files_velma = glob.glob(os.path.join(params.dir_pos_examples_velma, '*.jpg'))
with ThreadPoolExecutor() as executor:
    positive_patches_velma = list(executor.map(load_image, positive_files_velma))
positive_patches_velma = np.array(positive_patches_velma)
labels_velma = np.ones(len(positive_patches_velma), dtype=np.int64) * 2
print(f'Am încărcat {len(positive_patches_velma)} patch-uri Velma')

# --- Shaggy (label = 3) ---
positive_files_shaggy = glob.glob(os.path.join(params.dir_pos_examples_shaggy, '*.jpg'))
with ThreadPoolExecutor() as executor:
    positive_patches_shaggy = list(executor.map(load_image, positive_files_shaggy))
positive_patches_shaggy = np.array(positive_patches_shaggy)
labels_shaggy = np.ones(len(positive_patches_shaggy), dtype=np.int64) * 3
print(f'Am încărcat {len(positive_patches_shaggy)} patch-uri Shaggy')

# --- Daphne (label = 4) ---
positive_files_daphne = glob.glob(os.path.join(params.dir_pos_examples_daphne, '*.jpg'))
with ThreadPoolExecutor() as executor:
    positive_patches_daphne = list(executor.map(load_image, positive_files_daphne))
positive_patches_daphne = np.array(positive_patches_daphne)
labels_daphne = np.ones(len(positive_patches_daphne), dtype=np.int64) * 4
print(f'Am încărcat {len(positive_patches_daphne)} patch-uri Daphne')

# --- Negative (label = 0) ---
negative_files = glob.glob(os.path.join(params.dir_neg_examples, '*.jpg'))
with ThreadPoolExecutor() as executor:
    negative_patches = list(executor.map(load_image, negative_files))
negative_patches = np.array(negative_patches)
labels_negative = np.zeros(len(negative_patches), dtype=np.int64)  # Label = 0
print(f'Am încărcat {len(negative_patches)} patch-uri negative')

negative_files = glob.glob(os.path.join(params.dir_neg_examples_background, '*.jpg'))
with ThreadPoolExecutor() as executor:
    negative_patches_background = list(executor.map(load_image, negative_files))
negative_patches_background = np.array(negative_patches_background)
labels_negative_background = np.ones(len(negative_patches_background), dtype=np.int64) * 5  # Label = 5
print(f'Am încărcat {len(negative_patches_background)} patch-uri negative')





# --- Concatenare finală ---
training_examples = np.concatenate((
    positive_patches_fred,
    positive_patches_velma,
    positive_patches_shaggy,
    positive_patches_daphne,
    negative_patches,
    negative_patches_background,
), axis=0)

train_labels = np.concatenate((
    labels_fred,
    labels_velma,
    labels_shaggy,
    labels_daphne,
    labels_negative,
    labels_negative_background
))

print(f'\nTotal training examples: {len(training_examples)}')
print(f'  - Fred (label=1): {len(labels_fred)}')
print(f'  - Velma (label=2): {len(labels_velma)}')
print(f'  - Shaggy (label=3): {len(labels_shaggy)}')
print(f'  - Daphne (label=4): {len(labels_daphne)}')
print(f'  - Negative (label=0): {len(labels_negative)}')

print("====Incepe Antrenarea====")
facial_detector.train_classifier(training_examples, train_labels)

print("====Incepe Detectarea====")
results = facial_detector.run()

# Extrage datele pentru fiecare personaj
(detections_fred, scores_fred, file_names_fred), \
(detections_velma, scores_velma, file_names_velma), \
(detections_shaggy, scores_shaggy, file_names_shaggy), \
(detections_daphne, scores_daphne, file_names_daphne) = results

print("===Incepe Evaluare===")

salveaza_detectiile(detections_fred, scores_fred, file_names_fred, params, 'fred')
salveaza_detectiile(detections_velma, scores_velma, file_names_velma, params, 'velma')
salveaza_detectiile(detections_shaggy, scores_shaggy, file_names_shaggy, params, 'shaggy')
salveaza_detectiile(detections_daphne, scores_daphne, file_names_daphne, params, 'daphne')

if params.has_annotations:
    print(detections_fred)
    if detections_fred is not None:
        facial_detector.eval_detections_character(detections_fred, scores_fred, file_names_fred, 'task2_fred_gt_validare.txt', 'fred')
    print(detections_velma)
    if detections_velma is not None:
        facial_detector.eval_detections_character(detections_velma, scores_velma, file_names_velma,'task2_velma_gt_validare.txt', 'velma')
    print(detections_shaggy)
    if detections_shaggy is not None:
        facial_detector.eval_detections_character(detections_shaggy, scores_shaggy, file_names_shaggy,'task2_shaggy_gt_validare.txt', 'shaggy')
    print(detections_daphne)
    if detections_daphne is not None:
        facial_detector.eval_detections_character(detections_daphne, scores_daphne, file_names_daphne,'task2_daphne_gt_validare.txt', 'daphne')

    show_detections_with_ground_truth(detections_fred, scores_fred, file_names_fred, params, 'fred')
    show_detections_with_ground_truth(detections_velma, scores_velma, file_names_velma, params, 'velma')
    show_detections_with_ground_truth(detections_shaggy, scores_shaggy, file_names_shaggy, params, 'shaggy')
    show_detections_with_ground_truth(detections_daphne, scores_daphne, file_names_daphne, params, 'daphne')

