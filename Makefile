preprocess: preprocess.py
	python3 preprocess.py

matrix: matrix.py
	python3 matrix.py words.txt documents.txt

cluster: document_cluster.py word_cluster.py
	python3 document_cluster.py
	python3 word_cluster.py

clean:
	rm -rf words.txt documents.txt document_cluster.txt word_cluster.txt