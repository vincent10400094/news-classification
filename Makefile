preprocess: preprocess.py
	python3 preprocess.py

matrix: matrix.py
	python3 matrix.py words.txt documents.txt

clean:
	rm -rf words.txt documents.txt