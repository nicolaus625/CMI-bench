# # export PROJECT_ROOT=~

# cd $PROJECT_ROOT
# mkdir -p data/EMO
# cd data/EMO

wget http://cvml.unige.ch/databases/emoMusic/clips_45sec.tar.gz
wget http://cvml.unige.ch/databases/emoMusic/annotations.tar.gz
wget http://cvml.unige.ch/databases/emoMusic/dataset_manual.pdf

tar -xvf clips_45sec.tar.gz
tar -xvf annotations.tar.gz

# rm annotations.tar.gz 
# rm clips_45sec.tar.gz
# rm dataset_manual.pdf

python preprocess.py --dataset_dir .

# cd $PROJECT_ROOT