# baseline train
python train.py --name WAVLM --extractor WAVLM
python train.py --name STFT --extractor STFT
python train.py --name W2VL --extractor W2VL

# smote train
python train.py --smote --name WAVLM-SMOTE --extractor WAVLM
python train.py --smote --name STFT-SMOTE --extractor STFT
python train.py --smote --name W2VL-SMOTE --extractor W2VL
