from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

sc = SparkContext()
spark = SparkSession(sc)

# Read the data from BigQuery as a Spark Dataframe.
bank_data = spark.read.format("bigquery").option("table", "Bank.Defaulters").load()

df = bank_data.select("LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","payment")
cols = df.columns

stages = []

label_stringIdx = StringIndexer(inputCol = 'payment', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
assembler_Inputs = numericCols
assembler = VectorAssembler(inputCols=assembler_Inputs, outputCol="features")
stages += [assembler]


pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

train, test = df.randomSplit([0.8, 0.3], seed = 2018)
logistic = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
model_instance = logistic.fit(train)
model_instance.write().save("gs://automl-1//model")


# To generate predictions from model

predictions = model_instance.transform(test)
predictions.select("payment","prediction","probability").show(truncate=False)

predictions.show()


evaluator = BinaryClassificationEvaluator()
print('Area Under ROC', evaluator.evaluate(predictions))

predictions_data = predictions.select("label","prediction").rdd.map(tuple)
confusion_matrix = MulticlassMetrics(predictions_data)
print("recall: ",confusion_matrix.recall())
print("precision: ",confusion_matrix.precision())
