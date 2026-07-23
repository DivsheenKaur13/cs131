import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# A1 
spark = SparkSession.builder.appName("ws5-regression").getOrCreate()

# A2 
input_path = sys.argv[1]
df = spark.read.csv(input_path, header=True, inferSchema=True)
df.show()

# A3 
assembler = VectorAssembler(inputCols=["total_bill", "size"], outputCol="features")

# A4 
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# A5
lr = LinearRegression(featuresCol="features", labelCol="tip")
pipeline = Pipeline(stages=[assembler, lr])
pipelineModel = pipeline.fit(train_df)

# A6
predictions = pipelineModel.transform(test_df)

# A7
evaluator = RegressionEvaluator(labelCol="tip", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

# A8
lr_model = pipelineModel.stages[-1]
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

spark.stop()
