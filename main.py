# Imported Libraries:
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import argparse
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd


def processing_df(df):
    """
    This function cleans up the dataframe
    :param df: dataframe
    :return: dataframe
    """
    # Dropping Unnecessary Columns
    headers = df.columns
    for column_name in headers:
        if column_name != 'Date' \
                and column_name != 'Description' \
                and column_name != 'Amount' \
                and column_name != 'Class':
            df = df.drop(column_name)

    # Cleaning Date Column
    df = df.withColumn('Date', lpad(df['Date'], 10, '0')) \
        .withColumn('Date', to_date('Date', 'MM/dd/yyyy'))

    # Cleaning Description Column
    df = df.withColumn('Description', trim('Description')) \
        .withColumn('Description', regexp_replace('Description', "[^a-zA-Z]+", ' ')) \
        .withColumn('Description', lower('Description')) \
        .withColumn('Description', split(col('Description'), ' ')) \
        .withColumn('Description', when(size(expr("filter(Description, elem -> elem != '')")) == 0, lit(None))
                    .otherwise(expr("filter(Description, elem -> elem != '')")))

    # Cleaning Amount Column
    df = df.withColumn('Amount', regexp_replace('Amount', '[$(,)]', '')) \
        .withColumn("Amount", col("Amount").cast('double'))

    return df


def tdif_vectorization(df):
    """
    TDIF vectorization of Dataframe
    :param df: dataframe
    :return: dataframe
    """
    qualification_indexer = StringIndexer(inputCol='Class', outputCol='qualificationIndex')
    df1 = qualification_indexer.fit(df).transform(df)
    df1 = df1.withColumn('qualificationIndex', col('qualificationIndex').cast('int'))
    # df1.show()

    # HashingTF
    hashingTF = HashingTF(inputCol='Description', outputCol='rawFeatures')
    featureData = hashingTF.transform(df1)

    # IDF
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idfModel = idf.fit(featureData)
    rescaledData = idfModel.transform(featureData)
    new_data = rescaledData.select('qualificationIndex', 'features')
    # new_data.show()
    return new_data


def main():

    # Using arg parse for command line to read test file for classification
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, required=False)
    parser.add_argument('--p', type=bool, required=False)
    args = parser.parse_args()
    csv_filename = args.f
    if args.p == True:
        print('Yes')

    spark = SparkSession.builder \
        .appName("Bank Statement Classifier") \
        .master('local') \
        .getOrCreate()

    # Test Data for classification
    test_data = spark.read.option("delimiter", ",") \
        .option('inferSchema', 'True') \
        .option("header", "true") \
        .csv(csv_filename)

    # Training data for classification
    df = spark.read.option("delimiter", ",") \
        .option('inferSchema', 'True') \
        .option("header", "true") \
        .csv(csv_filename)

    df = processing_df(df)
    new_data = tdif_vectorization(df)

    # Splitting Data:
    splits = new_data.randomSplit([0.8, 0.2], 1234)
    train = splits[0]
    test = splits[1]
    # test.show()

    # Creating the trainer and set its parameters
    nb = NaiveBayes(labelCol='qualificationIndex', smoothing=1.0, modelType="multinomial")

    # Train the model
    model = nb.fit(train)
    # model.save(spark, 'naive_bayes.model')
    prediction = model.transform(test)
    prediction = prediction.withColumn('qualificationIndex', col('qualificationIndex').cast('double'))
    prediction.show()
    evaluator = MulticlassClassificationEvaluator(labelCol="qualificationIndex", predictionCol="prediction",
                                                  metricName="accuracy")

    accuracy = evaluator.evaluate(prediction)
    print("Test set accuracy = " + str(accuracy))

    spark.stop()


if __name__ == '__main__':
    main()
