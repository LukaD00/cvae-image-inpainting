# cvae-image-inpainting 🖌️✨

This repository is the implementation of cVAE model with GAN fine tuning used for the image inpainting task. It was created as part of a project for the Deep Learning 2 course under Faculty of Electrical Engineering and Engineering, Zagreb.

## Setup

1. Activate your virtual environment using Conda/venv.

2. ```bash
   pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 tqdm matplotlib notebook
   ```
   
## Train

1. Download aligned CelebA images and annotations from [this](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) link and save them following this hierarchy:
- cvae-image-inpainting
  - celeba
    - img_align_celeba *(folder with images)*
    - identity_CelebA.txt
    - list_attr_celeba.txt
    - list_bbox_celeba.txt
    - list_eval_partition.txt
    - list_landmarks_align_celeba.txt
    - list_landmarks_celeba.txt

2. Position yourself in the home project directory and run this to create the folder to store the weights.
```bash
   mkdir models/weights
 ```

3. Run the training script.
```py
  python main.py
  python fine_tune.py
```

## Inference

1. Download [our weights](https://drive.google.com/drive/folders/1I14RI2z_hnaW2NeCvHvUds2ogHgr06ZY) or train your own model and store them under `./model/weights`.
2. Open the `reconstruction_results.ipynb` notebook and run the cells to get inpainted images.
```bash
jupyter-notebook 
``` 

---

*Authors*:
- Tomislav Ćosić
- Luka Družijanić
- Renato Jurišić
- Marko Kremer
- Anto Matanović
- Josip Srzić
