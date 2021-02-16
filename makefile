all: install update-requirements prepare run

install:
	env/bin/pip install -r requirements.txt

update-requirements: install
	env/bin/pip freeze > requirements.txt

# data/prepared: data/data.csv
prepare:
	python src/prepare.py data/data.csv 

# final_model.pkl predicted/predictions.csv: data/prepared
run:	
	python src/star_type_predictions.py data/prepared final_model.pkl predicted/predictions.csv 