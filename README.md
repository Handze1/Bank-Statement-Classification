<h1> Pyspark Multinomial Classification Using Naive Bayes Classifier</h1>

- Columns of interest include: 
  - Date
  - Description
  - Amount
  - Class
- Using pyspark create a pipeline to clean -> vectorize -> classify statements. 

<h2>Setup for Users:</h2>

- Install pip-tools `pip install pip-tools`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install requirements `pip install -r requirements.txt`

<h2>Running Script:</h2>

- `python main.py --f <CSV File> --p True`


