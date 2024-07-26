import numpy as np


def rgb_to_2D_label(label, class_colors):
    label_seg = np.zeros(label.shape, dtype=np.uint8)
    for idx, color in enumerate(class_colors):
        label_seg[np.all(label == color, axis=-1)] = idx
    label_seg = label_seg[:, :, 0]
    return label_seg


Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i + 2], 16) for i in (0, 2, 4)))

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i + 2], 16) for i in (0, 2, 4)))

Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i + 2], 16) for i in (0, 2, 4)))

Vegetation = 'FEDD3A'.lstrip('#')
Vegetation = np.array(tuple(int(Vegetation[i:i + 2], 16) for i in (0, 2, 4)))

Water = 'E2A929'.lstrip('#')
Water = np.array(tuple(int(Water[i:i + 2], 16) for i in (0, 2, 4)))

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))

class_colors = [Building, Land, Road, Vegetation, Water, Unlabeled]
