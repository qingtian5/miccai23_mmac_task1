# MICCAI_TASK1

### Step 1 : Configure the mmpretrain environment

```bash
git clone https://github.com/LuliDreamAI/MICCAI_TASK1.git
cd MICCAI_TASK1
conda create -n miccai_mmpre python=3.8 pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
conda activate miccai_mmpre
pip3 install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
cd ..
```

### Step 2 : Place the images in the data directory

1. The directory under data should be as follows.

```bash
data
├── classification_train_val # This means the training set and the validation set are put together 
├── classification_train
└── classification_val
```

2. For each folder, the ground truth label file (meta/train.txt) needs to be generated in the following way. If it is a validation set, change the path to meta/val.txt.

```python
import pandas as pd

data = pd.read_csv("MICCAI_TASK1/data/classification_train/Groundtruths/train_labels.csv")

with open("MICCAI_TASK1/data/classification_train/meta/train.txt","w") as f:
    for idx, d in data.iterrows():
        f.write(d["image"] + " " + str(d["myopic_maculopathy_grade"]) + "\n")
```

3. The following format is required for each folder under data. If is the validation dataset, the label file should be meta/val.txt. 

```bash
classification_train
├── Groundtruths
│   └── train_labels.csv
├── Images
│   └── train
│       ├── mmac_task_1_train_0001.png
│       ├── mmac_task_1_train_0002.png
│       ├── ...
│       └── mmac_task_1_train_1143.png
├── LICENSE.txt
└── meta
    └── train.txt
```

```bash
classification_val
├── Groundtruths
│   └── val_labels.csv
├── Images
│   └── val
│       ├── mmac_task_1_val_0001.png
│       ├── mmac_task_1_val_0002.png
│       ├── ...
│       └── mmac_task_1_val_0248.png
├── LICENSE.txt
└── meta
    └── val.txt
```

### Step 3 : Place pre-trained weights in the pretrained_ckpt catalog.

Run the following command on the terminal

```bash
cd pretrained_ckpt
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth
cd ..
```

### Step 4 : Run the training script

Run the following command on the terminal

```bash
cd mmpretrain
python tools/train.py ../projects/submission/my_swin_base_in1k_384.py
```

### Step 5 : submit the results

Run the following command on the terminal

```bash
cd ../projects/submission
mv ../working/epoch_36.pth ./epoch_36.pth
zip -r submission.zip .
```

Then submit the submission.zip file to the competition website [Link](https://codalab.lisn.upsaclay.fr/competitions/12441#participate).

### TODO: Adding a weak label is complicated, and I'm going to release it quickly.


