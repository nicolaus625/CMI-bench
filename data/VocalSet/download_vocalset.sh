wget https://zenodo.org/record/1203819/files/VocalSet11.zip 
# unzip VocalSet11.zip
python preprocess.py
mv  data/VocalSet/* .
rm DataSetVocalises.pdf
rm VocalSet11.zip