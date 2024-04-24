pip install -r requirements.txt

mkdir video_data
cd video_data
gdown https://drive.google.com/uc?id=14JrtIbdo3xJU2o9wldbr-O2OK3VWR39B
unzip demo_videos.zip
rm demo_videos.zip
cd ..

# download pre-trained weights
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth