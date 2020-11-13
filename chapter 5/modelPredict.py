from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from  pyspark.ml.classification import LogisticRegressionModel
from pyspark.mllib.evaluation import MulticlassMetrics

sc = SparkContext()
spark = SparkSession(sc)

df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("gs://automl-1/prediction.csv")
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
model = LogisticRegressionModel.load("gs://automl-1/model")

predictions = model.transform(df)
predictions.select("prediction","probability").show(truncate=False)
predictions.show()


# To dump prediction result as csv into Google Cloud Storage bucket

predictions.withColumn("probability", col("probability").cast("string")).select("ID","prediction","probability").write.csv('gs://automl-1/prediction')



