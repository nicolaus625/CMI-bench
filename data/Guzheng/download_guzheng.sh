pip install modelscope
modelscope download --dataset ccmusic-database/Guzheng_Tech99
nv ~/.cache/modelscope/hub/datasets/ccmusic-database/Guzheng_Tech99 .

cd Guzheng_Tech99
unzip audio.zip
unzip label.zip
rm audio.zip label.zip mel.zip