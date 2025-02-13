# mkdir -p data/MTT
# cd data/MTT

wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003

cat mp3.zip* > mp3.zip
mkdir ./mp3
unzip mp3.zip -d ./mp3
rm mp3.*

wget https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv

cd ../..
python data/MTT/preprocess.py ./data
