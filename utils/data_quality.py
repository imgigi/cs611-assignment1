from pyspark.sql import functions as F
from pyspark.sql.types import *


def clean_data(df, table_name):
    if table_name == 'clickstream':
        return clean_clickstream(df)
    elif table_name == 'attributes':
        return clean_attributes(df)
    elif table_name == 'financials':
        return clean_financials(df)
    elif table_name == 'loans':
        return clean_loans(df)
    else:
        return df


def clean_clickstream(df):
    df = df.dropDuplicates(['Customer_ID', 'snapshot_date'])
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date'))
    
    feature_cols = [col for col in df.columns if col.startswith('fe_')]
    for col in feature_cols:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    
    return df


def clean_attributes(df):
    df = df.dropDuplicates(['Customer_ID', 'snapshot_date'])
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date'))
    
    df = df.withColumn('Age', F.regexp_replace('Age', '_', ''))
    df = df.withColumn('Age', F.col('Age').cast(IntegerType()))
    df = df.withColumn('Age', 
        F.when((F.col('Age') < 18) | (F.col('Age') > 100), None).otherwise(F.col('Age'))
    )
    df = df.withColumn('Age', F.coalesce(F.col('Age'), F.lit(35)))
    
    df = df.withColumn('SSN_valid', 
        F.when(F.col('SSN').rlike(r'^\d{3}-\d{2}-\d{4}$'), F.lit(1)).otherwise(F.lit(0))
    )
    
    df = df.withColumn('Occupation',
        F.when(F.col('Occupation') == '_______', 'Unknown').otherwise(F.col('Occupation'))
    )
    
    return df


def clean_financials(df):
    df = df.dropDuplicates(['Customer_ID', 'snapshot_date'])
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date'))
    
    df = df.withColumn('Annual_Income', F.regexp_replace('Annual_Income', '_', ''))
    df = df.withColumn('Annual_Income', F.col('Annual_Income').cast(DoubleType()))
    df = df.withColumn('Monthly_Inhand_Salary', F.col('Monthly_Inhand_Salary').cast(DoubleType()))
    
    df = df.withColumn('Num_of_Loan', F.regexp_replace('Num_of_Loan', '_', ''))
    df = df.withColumn('Num_of_Loan', F.col('Num_of_Loan').cast(IntegerType()))
    df = df.withColumn('Num_of_Loan', F.when(F.col('Num_of_Loan') > 15, 15).otherwise(F.col('Num_of_Loan')))
    
    df = df.withColumn('Num_Bank_Accounts', F.col('Num_Bank_Accounts').cast(IntegerType()))
    df = df.withColumn('Num_Bank_Accounts', F.coalesce(F.col('Num_Bank_Accounts'), F.lit(0)))
    
    df = df.withColumn('Num_Credit_Card', F.col('Num_Credit_Card').cast(IntegerType()))
    df = df.withColumn('Num_Credit_Card', F.when(F.col('Num_Credit_Card') > 20, 5).otherwise(F.col('Num_Credit_Card')))
    
    df = df.withColumn('Num_Credit_Inquiries', F.col('Num_Credit_Inquiries').cast(DoubleType()))
    df = df.withColumn('Num_Credit_Inquiries', F.when(F.col('Num_Credit_Inquiries') > 20, 20).otherwise(F.col('Num_Credit_Inquiries')))
    
    df = df.withColumn('Num_of_Delayed_Payment', F.regexp_replace('Num_of_Delayed_Payment', '_', ''))
    df = df.withColumn('Num_of_Delayed_Payment', F.col('Num_of_Delayed_Payment').cast(IntegerType()))
    
    df = df.withColumn('Credit_Mix', F.when(F.col('Credit_Mix') == '_', 'Unknown').otherwise(F.col('Credit_Mix')))
    df = df.withColumn('Payment_Behaviour', F.when(F.col('Payment_Behaviour').rlike(r'^[!@#\$%\^&\*\d]+$'), 'Unknown').otherwise(F.col('Payment_Behaviour')))
    
    df = df.withColumn('Amount_invested_monthly', F.regexp_replace('Amount_invested_monthly', '_', ''))
    df = df.withColumn('Amount_invested_monthly', F.col('Amount_invested_monthly').cast(DoubleType()))
    df = df.withColumn('Total_EMI_per_month', F.col('Total_EMI_per_month').cast(DoubleType()))
    df = df.withColumn('Changed_Credit_Limit', F.regexp_replace('Changed_Credit_Limit', '_', ''))
    df = df.withColumn('Changed_Credit_Limit', F.col('Changed_Credit_Limit').cast(DoubleType()))
    
    df = df.withColumn('Credit_History_Years', F.regexp_extract('Credit_History_Age', r'(\d+) Years', 1).cast(IntegerType()))
    df = df.withColumn('Credit_History_Months', F.regexp_extract('Credit_History_Age', r'(\d+) Months', 1).cast(IntegerType()))
    df = df.withColumn('Credit_History_Age_Months', F.col('Credit_History_Years') * 12 + F.col('Credit_History_Months'))
    df = df.drop('Credit_History_Age', 'Credit_History_Years', 'Credit_History_Months')
    
    df = df.withColumn('Payment_of_Min_Amount', F.when(F.col('Payment_of_Min_Amount') == 'NM', 'Unknown').otherwise(F.col('Payment_of_Min_Amount')))
    
    numeric_columns = ['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Monthly_Balance']
    for col in numeric_columns:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
    
    return df


def clean_loans(df):
    df = df.dropDuplicates(['loan_id', 'snapshot_date'])
    df = df.withColumn('loan_start_date', F.to_date('loan_start_date'))
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date'))
    
    numeric_columns = ['tenure', 'installment_num', 'loan_amt', 'due_amt', 'paid_amt', 'overdue_amt', 'balance']
    for col in numeric_columns:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
    
    amount_columns = ['due_amt', 'paid_amt', 'overdue_amt', 'balance']
    for col in amount_columns:
        if col in df.columns:
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))
    
    return df

