## RecCoder
This is the official PyTorch implementation for the paper:
> Kai-Huang Lai, Wu-Dong Xi, Xing-Xing Xing, Wei Wan, Chang-Dong Wang, Min Chen, Mohsen Guizani. "RecCoder: Reformulating Sequential Recommendation as Large Language Model-Based Code Completion". ICDM 2024.

### Requirements

```
torch==2.1.0
transformers==4.35.2
accelerate==0.23.0
scikit-learn==1.3.1
datasets==2.14.6
```

### Datasets

You can download the datasets using the following link: [Download Link](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G) from P5 repository.

```bash
unzip P5_data.zip
```

### Run

Beauty dataset as example

```
cd main_scripts
bash beauty_init_tokens.sh
bash beauty_training.sh
```

bash NYC_init_tokens.sh
bash NYC_training.sh