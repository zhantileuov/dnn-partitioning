import numpy as np


# The router keeps this mapping explicit on purpose:
# it avoids remote transfer of the large activation and lets us reuse one
# cached tensor per target model on the Triton host.
TARGET_MODEL_CATALOG = {
    "resnet18_tail_after_stem": {
        "shape": (1, 64, 56, 56),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer1_0": {
        "shape": (1, 64, 56, 56),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer1_1": {
        "shape": (1, 64, 56, 56),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer2_0": {
        "shape": (1, 128, 28, 28),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer2_1": {
        "shape": (1, 128, 28, 28),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer3_0": {
        "shape": (1, 256, 14, 14),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer3_1": {
        "shape": (1, 256, 14, 14),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer4_0": {
        "shape": (1, 512, 7, 7),
        "dtype": np.float32,
    },
    "resnet18_tail_after_layer4_1": {
        "shape": (1, 512, 7, 7),
        "dtype": np.float32,
    },
    "resnet50_tail_after_stem": {
        "shape": (1, 64, 56, 56),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer1_0": {
        "shape": (1, 256, 56, 56),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer1_1": {
        "shape": (1, 256, 56, 56),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer1_2": {
        "shape": (1, 256, 56, 56),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer2_0": {
        "shape": (1, 512, 28, 28),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer2_1": {
        "shape": (1, 512, 28, 28),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer2_2": {
        "shape": (1, 512, 28, 28),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer2_3": {
        "shape": (1, 512, 28, 28),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_0": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_1": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_2": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_3": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_4": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer3_5": {
        "shape": (1, 1024, 14, 14),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer4_0": {
        "shape": (1, 2048, 7, 7),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer4_1": {
        "shape": (1, 2048, 7, 7),
        "dtype": np.float32,
    },
    "resnet50_tail_after_layer4_2": {
        "shape": (1, 2048, 7, 7),
        "dtype": np.float32,
    },
}
