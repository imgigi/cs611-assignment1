import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

from utils.data_quality import clean_data
from utils.feature_engineering import create_financial_ratios, aggregate_clickstream_features
from utils.label_engineering import create_loan_labels


def init_spark():
    import os
    import sys
    
    # Fix for Windows (only needed when not in Docker)
    if sys.platform == 'win32':
        os.environ['HADOOP_HOME'] = 'C:\\hadoop'
    
    spark = SparkSession.builder \
        .appName("Loan Default Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def setup_directories():
    directories = ['datamart/bronze', 'datamart/silver', 'datamart/gold']
    
    if os.path.exists('datamart'):
        shutil.rmtree('datamart')
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created")


def bronze_layer(spark):
    # Bronze layer - load raw data
    print("\n[Bronze Layer] Loading raw data...")
    
    df_clickstream = spark.read.csv('data/feature_clickstream.csv', header=True, inferSchema=True)
    df_clickstream.write.mode('overwrite').parquet('datamart/bronze/clickstream')
    print(f"Clickstream: {df_clickstream.count()} rows")
    
    df_attributes = spark.read.csv('data/features_attributes.csv', header=True, inferSchema=True)
    df_attributes.write.mode('overwrite').parquet('datamart/bronze/attributes')
    print(f"Attributes: {df_attributes.count()} rows")
    
    df_financials = spark.read.csv('data/features_financials.csv', header=True, inferSchema=True)
    df_financials.write.mode('overwrite').parquet('datamart/bronze/financials')
    print(f"Financials: {df_financials.count()} rows")
    
    df_loans = spark.read.csv('data/lms_loan_daily.csv', header=True, inferSchema=True)
    df_loans.write.mode('overwrite').parquet('datamart/bronze/loans')
    print(f"Loans: {df_loans.count()} rows")
    
    return df_clickstream, df_attributes, df_financials, df_loans


def silver_layer(spark):
    # Silver layer - clean and validate data
    print("\n[Silver Layer] Cleaning data...")
    
    df_clickstream = spark.read.parquet('datamart/bronze/clickstream')
    df_attributes = spark.read.parquet('datamart/bronze/attributes')
    df_financials = spark.read.parquet('datamart/bronze/financials')
    df_loans = spark.read.parquet('datamart/bronze/loans')
    
    df_clickstream_clean = clean_data(df_clickstream, 'clickstream')
    df_clickstream_clean.write.mode('overwrite').parquet('datamart/silver/clickstream')
    
    df_attributes_clean = clean_data(df_attributes, 'attributes')
    df_attributes_clean.write.mode('overwrite').parquet('datamart/silver/attributes')
    
    df_financials_clean = clean_data(df_financials, 'financials')
    df_financials_clean.write.mode('overwrite').parquet('datamart/silver/financials')
    
    df_loans_clean = clean_data(df_loans, 'loans')
    df_loans_clean.write.mode('overwrite').parquet('datamart/silver/loans')
    
    print("Data cleaning complete")
    return df_clickstream_clean, df_attributes_clean, df_financials_clean, df_loans_clean


def gold_layer(spark):
    # Gold layer - create feature and label stores
    print("\n[Gold Layer] Building feature and label stores...")
    
    df_clickstream = spark.read.parquet('datamart/silver/clickstream')
    df_attributes = spark.read.parquet('datamart/silver/attributes')
    df_financials = spark.read.parquet('datamart/silver/financials')
    df_loans = spark.read.parquet('datamart/silver/loans')
    
    # Create label store
    df_label_store = create_loan_labels(df_loans)
    df_label_store.write.mode('overwrite').parquet('datamart/gold/label_store')
    print(f"Label store: {df_label_store.count()} loan applications")
    
    # Get loan applications
    loan_applications = df_loans.filter(F.col('installment_num') == 0).select(
        'Customer_ID',
        F.col('loan_start_date').alias('application_date')
    ).distinct()
    
    # Process clickstream features - only use data before application date
    df_clickstream_filtered = df_clickstream.join(
        loan_applications, on='Customer_ID', how='inner'
    ).filter(F.col('snapshot_date') <= F.col('application_date'))
    
    window_clickstream = Window.partitionBy('Customer_ID', 'application_date').orderBy(F.desc('snapshot_date'))
    df_clickstream_features = df_clickstream_filtered.withColumn(
        'rank', F.row_number().over(window_clickstream)
    ).filter(F.col('rank') == 1).drop('rank', 'snapshot_date')
    
    df_clickstream_features = aggregate_clickstream_features(df_clickstream_features)
    
    # Process customer attributes
    df_attributes_filtered = df_attributes.join(
        loan_applications, on='Customer_ID', how='inner'
    ).filter(F.col('snapshot_date') <= F.col('application_date'))
    
    window_attributes = Window.partitionBy('Customer_ID', 'application_date').orderBy(F.desc('snapshot_date'))
    df_attributes_features = df_attributes_filtered.withColumn(
        'rank', F.row_number().over(window_attributes)
    ).filter(F.col('rank') == 1).drop('rank', 'snapshot_date', 'Name', 'SSN')
    
    # Process financial features
    df_financials_filtered = df_financials.join(
        loan_applications, on='Customer_ID', how='inner'
    ).filter(F.col('snapshot_date') <= F.col('application_date'))
    
    window_financials = Window.partitionBy('Customer_ID', 'application_date').orderBy(F.desc('snapshot_date'))
    df_financials_features = df_financials_filtered.withColumn(
        'rank', F.row_number().over(window_financials)
    ).filter(F.col('rank') == 1).drop('rank', 'snapshot_date')
    
    df_financials_features = create_financial_ratios(df_financials_features)
    
    # Combine all features
    df_feature_store = df_clickstream_features \
        .join(df_attributes_features, on=['Customer_ID', 'application_date'], how='inner') \
        .join(df_financials_features, on=['Customer_ID', 'application_date'], how='inner')
    
    df_feature_store = df_feature_store.withColumnRenamed('application_date', 'loan_start_date')
    
    df_feature_store.write.mode('overwrite').parquet('datamart/gold/feature_store')
    print(f"Feature store: {df_feature_store.count()} records, {len(df_feature_store.columns)-2} features")
    
    # Create ML ready dataset
    df_ml_ready = df_feature_store.join(
        df_label_store, on=['Customer_ID', 'loan_start_date'], how='inner'
    )
    df_ml_ready.write.mode('overwrite').parquet('datamart/gold/ml_ready_dataset')
    print(f"ML-ready dataset: {df_ml_ready.count()} records")
    
    return df_feature_store, df_label_store, df_ml_ready


def main():
    print("\nLoan Default Prediction - Data Pipeline")
    print("="*50)
    
    spark = init_spark()
    setup_directories()
    
    bronze_layer(spark)
    silver_layer(spark)
    gold_layer(spark)
    
    print("\n" + "="*50)
    print("Pipeline execution complete")
    print("="*50)
    
    spark.stop()


if __name__ == "__main__":
    main()

