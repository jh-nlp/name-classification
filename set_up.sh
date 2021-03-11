sudo apt-get install python-numpy libicu-dev
python -m spacy download en

wget http://nlp.stanford.edu/data/glove.6B.zip data/glove.6B.zip
unzip glove.6B.zip -d /home/jupyter/sb-entity-classification/data/glove.6B
rm glove.6B.zip