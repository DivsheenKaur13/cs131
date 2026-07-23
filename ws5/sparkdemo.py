import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# A1 — SparkSession
spark = SparkSession.builder.appName("ws5-regression").getOrCreate()

# A2 — read from bucket, path passed as command-line arg
input_path = sys.argv[1]
df = spark.read.csv(input_path, header=True, inferSchema=True)
df.show()

# A3 — combine predictors into a single "features" vector column
assembler = VectorAssembler(inputCols=["total_bill", "size"], outputCol="features")

# A4 — 80/20 split, reproducible with a seed
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# A5 — define the regressor, chain into a Pipeline, fit on training data
lr = LinearRegression(featuresCol="features", labelCol="tip")
pipeline = Pipeline(stages=[assembler, lr])
pipelineModel = pipeline.fit(train_df)

# A6 — apply the fitted pipeline to the test set
predictions = pipelineModel.transform(test_df)

# A7 — evaluate RMSE and R²
evaluator = RegressionEvaluator(labelCol="tip", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = evaluator.setMetricName("r2").evaluate(predictions)

# A8 — pull the fitted LinearRegression model out of the pipeline, print everything clearly
lr_model = pipelineModel.stages[-1]
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

spark.stop()
