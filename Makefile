PRIVATE_KEY_PATH =
USERNAME =

install:
	pip install -r requirements.txt

#download_dataset:
#	mkdir dataset
#	rsync -aSvuc -e 'ssh -p 22022 -i $(PRIVATE_KEY_PATH)' $(USERNAME)@ml-server.avtodoria.ru:/home/alik/Snow/dataset/ dataset/

lint:
	PYTHONPATH=. flake8 src

clean-logs: ## Clean logs
	rm -rf logs/**

train: ## Train the model
	python src/train.py

test:
	python src/eval.py
