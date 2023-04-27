<h1> Pyspark Multinomial Classification Using Naive Bayes Classifier</h1>

- Columns of interest include: 
  - Date
  - Description
  - Amount
  - Class
- Using pyspark, create a pipeline to clean -> vectorize -> classify statements. 
  - Cleaning: Dropping unneccessary columns, converting to Datetime, removing special characters/numbers in description and converting amount to type double
  - Vectorize: Use TF-IDF, to mine text in the description column
  - Classification: Using Naive Bayes Model to classify multinomial statements
  

<h2>Setup for Users:</h2>

- Install pip-tools `pip install pip-tools`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install requirements `pip install -r requirements.txt`

<h2>Running Script:</h2>

- `python main.py --f <Training CSV> --p <Testing CSV>`
- --f is the training file 
- --p is the testing file for prediction


-------
Output:

![image](https://user-images.githubusercontent.com/101682684/234975681-7a6068af-6db6-4bdc-babf-4030c712c263.png)

![image](https://user-images.githubusercontent.com/101682684/234975842-26396403-5298-4c4e-8392-ca6ec615b9ec.png)


