mkdir data
cd data

echo "Downloading datasets..."
gdown https://drive.google.com/uc?id=12IukAqxOj497J4F0Mel-FvaONM030qwP
gdown https://drive.google.com/uc?id=1U0MZVro5yGgaHNQ8kWb3U1a0Qlz4HiHI
gdown https://drive.google.com/uc?id=1cLkZs8pN8onQzfyNskeU_836JLjrtJz1
gdown https://drive.google.com/uc?id=19Kd8jJLxZExwiTc9__6_r_jPqkmTXt4h
gdown https://drive.google.com/uc?id=13UidsM1taqEAVV_JJTBiCV1D3KUBpmpj
gdown https://drive.google.com/uc?id=1f4sm8hpWQRzZMpvM-j7Q3xPG2vtdwvTy
gdown https://drive.google.com/uc?id=1em51XXz5_aBNRJlJxxv3-Ed1dO9H3QgS


echo "Unzipping datasets... (this may take a while)"
echo "Unzipping MVTEC AD"
unzip -qq mvtec_anomaly_detection.zip -d mvtec_anomaly_detection

echo "Unzipping VisA"
unzip -qq VisA_20220922.zip -d VisA_20220922

echo "Unzipping MPDD"
unzip -qq MPDD.zip -d MPDD

echo "Unzipping BTech"
unzip -qq BTech_Dataset_transformed.zip -d BTech_Dataset_transformed

echo "Unzipping SDD"
unzip -qq SDD_anomaly_detection.zip -d SDD_anomaly_detection

echo "Unzipping DAGM"
unzip -qq DAGM_anomaly_detection.zip -d DAGM_anomaly_detection

echo "Unzipping DTD"
unzip -qq DTD-Synthetic.zip -d DTD-Synthetic

echo "Done!"

cd ..