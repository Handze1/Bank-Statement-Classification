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
    :param df: Dataframe
    :return: Dataframe
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
    df = df.withColumn('Date', to_date('Date', 'M/d/yyyy'))

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


def string_indexer(df):
    """
    Label indexer that maps a string column "Class" to label indices
    :param df: Dataframe
    :return: Dataframe
    """
    qualification_indexer = StringIndexer(inputCol='Class', outputCol='qualificationIndex')
    df = qualification_indexer.fit(df).transform(df)
    df = df.withColumn('qualificationIndex', col('qualificationIndex').cast('int'))
    return df


def tdif_vectorization(df):
    """
    TDIF vectorization of Dataframe
    :param df: dataframe
    :return: dataframe
    """
    # HashingTF
    hashingTF = HashingTF(inputCol='Description', outputCol='rawFeatures')
    featureData = hashingTF.transform(df)

    # IDF
    idf = IDF(inputCol='rawFeatures', outputCol='features')
    idfModel = idf.fit(featureData)
    rescaledData = idfModel.transform(featureData)
    new_data = rescaledData.select('*')
    # new_data.show()
    return new_data


def main():
    # Using arg parse for command line to read test file for classification
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, required=False)
    parser.add_argument('--p', type=str, required=False)
    args = parser.parse_args()

    # Creating Spark Session
    spark = SparkSession.builder \
        .appName("Bank Statement Classifier") \
        .master('local') \
        .getOrCreate()

    # Reading in Training Data Frame
    train_data = spark.read.option("delimiter", ",") \
        .option('inferSchema', 'True') \
        .option("header", "true") \
        .csv(args.f)

    # Reading in Testing Data Frame
    test_data = spark.read.option("delimiter", ",") \
        .option('inferSchema', 'True') \
        .option("header", "true") \
        .csv(args.p)

    # Checking argparse arguments
    if type(args.f) is str and type(args.p) is str:
        # Calling Processing, string indexer, TDIF vectorization function on Training/Testing Data
        # Testing Data does not Require string indexer because data has no class.
        df_train = processing_df(train_data)
        df_train = string_indexer(df_train)
        df_train = tdif_vectorization(df_train)
        df_test = processing_df(test_data)
        df_test = tdif_vectorization(df_test)
        print('Training and Testing Data Cleaned')

        # Splitting Training Data to metrics
        splits = df_train.randomSplit([0.8, 0.2], 1234)
        train = splits[0]
        test = splits[1]

        # Creating the trainer and set its parameters
        nb = NaiveBayes(labelCol='qualificationIndex', smoothing=1.0, modelType="multinomial")

        # Training model
        model = nb.fit(train)
        prediction = model.transform(test)
        prediction = prediction.withColumn('qualificationIndex', col('qualificationIndex').cast('double'))
        # prediction.show()

        # Evaluation
        acc = MulticlassClassificationEvaluator(labelCol="qualificationIndex", predictionCol="prediction",
                                                metricName="accuracy")
        f1 = MulticlassClassificationEvaluator(labelCol="qualificationIndex", predictionCol="prediction",
                                               metricName="f1")
        precision = MulticlassClassificationEvaluator(labelCol="qualificationIndex", predictionCol="prediction",
                                                      metricName="precisionByLabel")

        accuracy = acc.evaluate(prediction)
        F1 = f1.evaluate(prediction)
        Precision = precision.evaluate(prediction)

        print("Test set accuracy = " + str(accuracy))
        print("Test set f1 = " + str(F1))
        print("Test set Precision = " + str(Precision))

        # Predicting test file using trained model
        predicting_test = model.transform(df_test)
        predicting_test.show()

    spark.stop()


if __name__ == '__main__':
    main()
