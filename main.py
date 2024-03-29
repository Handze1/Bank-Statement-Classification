# Imported Libraries:
import argparse
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import plotly.express as px


mapping = {0: 'Alcohol', 1: 'Food', 2: 'Other', 3: 'Venmo', 4: 'Ride',
           5: 'Subscription', 6: 'Gas', 7: 'Healthcare', 8: 'Deposit',
           9: 'Investments', 10: 'School'}


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

    # Checking argparse arguments
    if type(args.f) is str and type(args.p) is str:
        train_data = spark.read.option("delimiter", ",") \
            .option('inferSchema', 'True') \
            .option("header", "true") \
            .csv(args.f)

        # Reading in Test Data Frame
        test_data = spark.read.option("delimiter", ",") \
            .option('inferSchema', 'True') \
            .option("header", "true") \
            .csv(args.p)

        # Calling Processing, string indexer, TDIF vectorization function on Data frames
        # Testing Data does not require string indexer because data has no class.
        df_train = processing_df(train_data)
        df_train = string_indexer(df_train)
        df_train = tdif_vectorization(df_train)
        df_test = processing_df(test_data)
        df_test = tdif_vectorization(df_test)
        print('Data Cleaned')

        # Splitting Training Data for model Evaluation
        splits = df_train.randomSplit([0.8, 0.2], 1234)
        train = splits[0]
        test = splits[1]

        # Creating the trainer and set its parameters
        nb = NaiveBayes(labelCol='qualificationIndex', smoothing=1.0, modelType="multinomial")

        # Training model
        model = nb.fit(train)

        # Saving/Overwriting Model
        model.write().overwrite().save("/NB_model")
        print('Model Saved')

        # Prediction based on test
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
        # predicting_test.show()

        predictionsDF = predicting_test.withColumn('prediction', col('prediction').cast('int'))
        predictionsDF.show()

        # Converting spark DF to pandas DF
        predictionsDF = predictionsDF.toPandas()
        spark.stop()

        # Removing unnecessary columns
        predictionsDF = predictionsDF.drop('rawFeatures', axis=1) \
            .drop('features', axis=1) \
            .drop('rawPrediction', axis=1) \
            .drop('probability', axis=1)

        # Mapping int predictions to string
        predictionsDF = predictionsDF.replace({'prediction': mapping})

        # # Spending Overview:
        # # Pie Chart Figure Without Deposits
        spending = predictionsDF[predictionsDF.prediction != 'Deposit']
        # pie_chart = spending.groupby(["prediction"]).sum()
        pie_fig = px.pie(spending, values='Amount', names='prediction', title='Breakdown of Spending')
        pie_fig.show()

        # Line chart
        line_chart = predictionsDF[predictionsDF.prediction != 'Deposit'] \
            .drop('prediction', axis=1) \
            .groupby(['Date']).sum()
        line_chart = px.line(line_chart, y="Amount", title='Spending Over Time')
        line_chart.show()

    # If --p is not a string, it will just train the model and produce evaluation results
    elif type(args.p) != str:
        train_data = spark.read.option("delimiter", ",") \
            .option('inferSchema', 'True') \
            .option("header", "true") \
            .csv(args.f)

        df_train = processing_df(train_data)
        df_train = string_indexer(df_train)
        df_train = tdif_vectorization(df_train)
        print('Data Cleaned')

        # Splitting Training Data to metrics
        splits = df_train.randomSplit([0.8, 0.2], 1234)
        train = splits[0]
        test = splits[1]

        # Creating the trainer and set its parameters
        nb = NaiveBayes(labelCol='qualificationIndex', smoothing=1.0, modelType="multinomial")

        # Training model
        model = nb.fit(train)

        # Saving/overwriting model
        model.write().overwrite().save("NB_model")
        print('Model Saved')

        # Prediction for training data for evaluation of model
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

        print("Naive Bayes Model has been trained!")
        print("Test set accuracy = " + str(accuracy))
        print("Test set f1 = " + str(F1))
        print("Test set Precision = " + str(Precision))
        spark.stop()

    elif type(args.f) != str:
        # Checking if model exists
        if os.path.isdir('NB_model') is True:
            # If model exists read in new test data
            test_data = spark.read.option("delimiter", ",") \
                .option('inferSchema', 'True') \
                .option("header", "true") \
                .csv(args.p)

            # Cleaning New testing data.
            df_test = processing_df(test_data)
            df_test = tdif_vectorization(df_test)
            print("Data Cleaned")
            # df_test.show()

            # Loading NB_model
            loaded_model = NaiveBayesModel.load('NB_model')

            # Classifying new data
            predictionsDF = loaded_model.transform(df_test)
            predictionsDF = predictionsDF.withColumn('prediction', col('prediction').cast('int'))
            predictionsDF.show()

            # Converting spark DF to pandas DF
            predictionsDF = predictionsDF.toPandas()
            spark.stop()

            predictionsDF = predictionsDF.drop('rawFeatures', axis=1) \
                .drop('features', axis=1) \
                .drop('rawPrediction', axis=1) \
                .drop('probability', axis=1)

            predictionsDF = predictionsDF.replace({'prediction': mapping})
            # print(predictionsDF)

            # Spending Overview:
            # Pie Chart Figure Without Deposits
            spending = predictionsDF[predictionsDF.prediction != 'Deposit']
            pie_fig = px.pie(spending, values='Amount', names='prediction', title='Breakdown of Spending')
            pie_fig.show()

            # Line chart: Spending over time
            line_chart = predictionsDF[predictionsDF.prediction != 'Deposit'] \
                .drop('prediction', axis=1) \
                .groupby(['Date']).sum()
            line_chart = px.line(line_chart, y="Amount", title='Spending Over Time')
            line_chart.show()

        else:
            print('No Training Model')
            spark.stop()


if __name__ == '__main__':
    main()
