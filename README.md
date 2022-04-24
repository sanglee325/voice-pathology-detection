# Voice Pathology Detection

## Environment Setup

* Create conda environment.

    ```bash
    conda create -n vpd python=3.9
    conda activate vpd
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Dataset

* Features must be prepared in below file structure.
    
    ```bash
    voice-pathology-detection
        ├── data
        │   ├── STFT
        │   │   ├── train
        │   │   ├── val
        │   │   └── label
        │   ├── W2VL
        │   │   ├── train
        │   │   ├── val
        │   │   └── label
        │   └── WAVLM
        │       ├── train
        │       ├── val
        │       └── label
        │ 
        ├── ...
    ```

* Features can be divided by using `divide_dataset.py`.

    ```bash
    python divide_dataset.py --path <feature_dir>
    ```

## Running Code

*   ```bash
    # baseline train
    python train.py --extractor <extractor_type>

    # train using smote
    python train.py --smote --extractor <extractor_type>
    ```