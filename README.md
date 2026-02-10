# 🔍 Scooby-Doo Face Detector

Proiect CNN pentru detectarea și clasificarea fețelor personajelor din Scooby-Doo folosind deep learning.

## 📋 Descriere

Acest proiect implementează un sistem de detecție și clasificare a fețelor personajelor din desenul animat Scooby-Doo folosind rețele neuronale convoluționale (CNN). Proiectul este împărțit în trei task-uri principale:

- **Task 1**: Detecție binară (față/non-față)
- **Task 2**: Clasificare multi-clasă (5 personaje + background)
- **Bonus**: Implementare YOLO pentru performanță îmbunătățită

## 🎯 Task 1: Detecția Fețelor

### Pregătirea Datelor

- **Exemple pozitive**: Imagini cropate și redimensionate la 64×84 pixeli folosind algoritmul Lanczos4
- **Exemple negative**: 45,000 de patch-uri generate aleator cu IoU ≤ 30%
- **Augmentare date**: Rotații aleatorii (±10°) și ajustări de luminozitate/contrast

```python
augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

### Arhitectura Modelului

Rețea CNN cu 3 straturi convoluționale:

1. **Conv1**: 3 → 32 canale (extrage trăsături primare)
2. **Conv2**: 32 → 64 canale (rafineză trăsături complexe)
3. **Conv3**: 64 → 128 canale (abstractizează trăsături)

**Regularizare**:
- Dropout 0.25 pe features
- Dropout 0.5 pe classifier

**Funcții de activare**:
- ReLU pentru straturi ascunse
- Sigmoid pentru output

### Antrenare

- **Optimizator**: Adam (learning rate: 1e-3)
- **Loss function**: Binary Cross Entropy
- **Detecție**: Multiscale sliding window

### Performanță

- ✅ **Acuratețe validare**: 75%
- ✅ **Average Precision**: 79%

## 🎭 Task 2: Clasificarea Personajelor

### Clase

```
0 – Unknown
1 – Fred
2 – Velma
3 – Shaggy
4 – Daphne
5 – Background
```

### Pregătirea Datelor

Augmentare diferențiată:

**Pentru fețe** (agresivă):
```python
face_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((64, 84)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Pentru background/unknown** (moderată):
```python
unknown_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(64, 84), scale=(0.9, 1.0)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

### Arhitectura Modelului

CNN cu 6 clase de output și regularizare weight decay.

### Antrenare

- **Optimizator**: Adam (weight decay: 1e-4)
- **Detecție**: Multiscale sliding window + Non-Maximum Suppression per label

### Performanță

- ✅ **Average Precision Mean**: 55%

## 🚀 Bonus: Implementare YOLO

Am antrenat un detector YOLO custom pe dataset-ul generat:

- **Epoci**: 15
- **Rezultat**: Average Precision îmbunătățit față de CNN standard pentru ambele task-uri

## 🛠️ Tehnologii Utilizate

- **Python 3.x**
- **PyTorch** - Framework deep learning
- **OpenCV** - Procesare imagini
- **torchvision** - Transformări și augmentare
- **NumPy** - Operații numerice

## 📦 Instalare

```bash
pip install torch torchvision opencv-python numpy
```

## 🎮 Utilizare

### Task 1 - Detecție Binară
```bash
python task1.py --input <imagine> --output <rezultat>
```

### Task 2 - Clasificare Multi-clasă

- ✅ **Acuratețe validare**: 98%
- ✅ **Average Precision**: 96%
```bash
python task2.py --input <imagine> --output <rezultat>
```

### Bonus - YOLO
- ✅ **Acuratețe validare**: 96%
- ✅ **Average Precision**: 95%
```bash
python bonus.py --input <imagine> --output <rezultat>
```

## 📊 Structura Proiectului

```
Scoobydoo-face-detector/
├── Face_detector/
│   ├── antrenare/          # Script-uri de antrenare
│   ├── solutie_task1/      # Implementare Task 1
│   ├── solutie_task2/      # Implementare Task 2
│   ├── solutie_bonus/      # Implementare YOLO
│   └── testare/            # Script-uri de testare
├── README.md
└── requirements.txt
```

## 📝 Note Tehnice

### Prevenirea Overfitting-ului

Problema inițială de overfitting (acuratețe mare pe training, scăzută pe validare) a fost rezolvată prin:
- Aplicarea dropout (0.25 și 0.5)
- Augmentare extensivă a datelor
- Weight decay pentru regularizare

### Optimizări

- **Lanczos4**: Minimizează distorsiunea la redimensionare
- **IoU Threshold**: 30% pentru exemple negative asigură diversitate
- **NMS**: Elimină detecții duplicate per clasă

## 👨‍💻 Autor

Luminararu Ionut

## 📄 Licență

Acest proiect a fost dezvoltat ca parte a unui curs academic.

---

**Scooby-Doo and the Mystery of Deep Learning! 🐕**
