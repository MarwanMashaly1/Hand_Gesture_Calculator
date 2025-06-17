import os
import numpy as np

DATA_DIR = 'data/raw_landmarks'
GESTURES = sorted(os.listdir(DATA_DIR))  # folder names as labels

X = []
y = []

for label, gesture in enumerate(GESTURES):
    folder = os.path.join(DATA_DIR, gesture)
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            X.append(np.load(os.path.join(folder, file)))
            y.append(label)

X = np.array(X)
y = np.array(y)
np.save('X.npy', X)
np.save('y.npy', y)
print("Dataset prepared:", X.shape, y.shape)

