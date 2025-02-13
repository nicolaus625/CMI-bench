export PROJECT_ROOT=/import/c4dm-datasets-ext/yinghao_tmp/CMI-bench

cd $PROJECT_ROOT

# mkdir -p data/GS
cd data/GS-key
wget https://github.com/GiantSteps/giantsteps-mtg-key-dataset/archive/fd7b8c584f7bd6d720d170c325a6d42c9bf75a6b.zip -O mtg-key-annotations.zip
wget https://github.com/GiantSteps/giantsteps-key-dataset/archive/c8cb8aad2cb53f165be51ea099d0dc75c64a844f.zip -O key-annotations.zip

cd $PROJECT_ROOT
python data/GS-key/download.py --data_path .

cd data/GS-key
unzip key-annotations.zip
unzip mtg-key-annotations.zip
rm mtg-key-annotations.zip key-annotations.zip

cd ../..
python data/GS-key/preprocess.py --dataset_dir data/GS-key
