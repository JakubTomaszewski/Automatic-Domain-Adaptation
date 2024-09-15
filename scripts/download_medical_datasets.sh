mkdir data
cd data

echo "Downloading datasets..."
gdown https://drive.google.com/uc?id=1ore0yCV31oLwwC--YUuTQfij-f2V32O2
gdown https://drive.google.com/uc?id=1JLYyzcPG3ULY2J_aw1SY9esNujYm9GKd

echo "Unzipping datasets... (this may take a while)"
echo "Unzipping HeadCT"
unzip -qq HeadCT_anomaly_detection.zip -d HeadCT_anomaly_detection

echo "Unzipping VisA"
unzip -qq BrainMRI.zip -d BrainMRI

echo "Done!"

cd ..