init:
	pip install -r requirements.txt
	mkdir data   
	python -m spacy download en_core_web_md 
	wget http://nlp.stanford.edu/data/glove.6B.zip data/glove.6B.zip
	unzip glove.6B.zip -d /home/jupyter/sb-entity-classification/data/glove.6B
	rm glove.6B.zip
