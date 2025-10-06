from pyspark.sql import functions as F
from pyspark.sql.types import *


def aggregate_clickstream_features(df):
    # Get clickstream features for each customer at application time
    feature_cols = [col for col in df.columns if col.startswith('fe_')]
    return df.select('Customer_ID', 'application_date', *feature_cols)


def create_financial_ratios(df):
    # Add derived financial features
    df = df.withColumn('debt_to_income_ratio', 
        F.when(F.col('Annual_Income') > 0, F.col('Outstanding_Debt') / F.col('Annual_Income')).otherwise(0))
    
    df = df.withColumn('loan_to_income_ratio',
        F.when(F.col('Annual_Income') > 0, F.col('Total_EMI_per_month') * 12 / F.col('Annual_Income')).otherwise(0))
    
    df = df.withColumn('investment_to_income_ratio',
        F.when(F.col('Annual_Income') > 0, F.col('Amount_invested_monthly') * 12 / F.col('Annual_Income')).otherwise(0))
    
    df = df.withColumn('balance_to_salary_ratio',
        F.when(F.col('Monthly_Inhand_Salary') > 0, F.col('Monthly_Balance') / F.col('Monthly_Inhand_Salary')).otherwise(0))
    
    df = df.withColumn('payment_stress_ratio',
        F.when(F.col('Monthly_Inhand_Salary') > 0, F.col('Total_EMI_per_month') / F.col('Monthly_Inhand_Salary')).otherwise(0))
    
    return df

