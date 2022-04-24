# baseline train
python train.py --name WAVLM --extractor WAVLM --arch cnn
python train.py --name WAVLM --extractor WAVLM --arch resnet18 
python train.py --name WAVLMP --extractor WAVLM --arch resnet18 -p
python train.py --name WAVLM --extractor WAVLM --arch lstm

python train.py --name WVLML --extractor WVLML --arch cnn
python train.py --name WVLML --extractor WVLML --arch resnet18 
python train.py --name WVLMLP --extractor WVLML --arch resnet18 -p
python train.py --name WVLML --extractor WVLML --arch lstm

python train.py --name STFT --extractor STFT --arch cnn
python train.py --name STFT --extractor STFT --arch resnet18
python train.py --name STFTP --extractor STFT --arch resnet18 -p
python train.py --name STFT --extractor STFT --arch lstm

python train.py --name W2VL --extractor W2VL --arch cnn
python train.py --name W2VL --extractor W2VL --arch resnet18
python train.py --name W2VLP --extractor W2VL --arch resnet18 -p
python train.py --name W2VL --extractor W2VL --arch lstm

# smote train
python train.py --smote --name WAVLM-S --extractor WAVLM --arch cnn
python train.py --smote --name WAVLM-S --extractor WAVLM --arch resnet18
python train.py --smote --name WAVLMP-S --extractor WAVLM --arch resnet18 -p
python train.py --smote --name WAVLM-S --extractor WAVLM --arch lstm

python train.py --smote --name WVLML-S --extractor WVLML --arch cnn
python train.py --smote --name WVLML-S --extractor WVLML --arch resnet18
python train.py --smote --name WVLMLP-S --extractor WVLML --arch resnet18 -p
python train.py --smote --name WVLML-S --extractor WVLML --arch lstm

python train.py --smote --name STFT-S --extractor STFT --arch cnn
python train.py --smote --name STFT-S --extractor STFT --arch resnet18
python train.py --smote --name STFTP-S --extractor STFT --arch resnet18 -p
python train.py --smote --name STFT-S --extractor STFT --arch lstm

python train.py --smote --name W2VL-S --extractor W2VL --arch cnn
python train.py --smote --name W2VL-S --extractor W2VL --arch resnet18
python train.py --smote --name W2VLP-S --extractor W2VL --arch resnet18 -p
python train.py --smote --name W2VL-S --extractor W2VL --arch lstm
