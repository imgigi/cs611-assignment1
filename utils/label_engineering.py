from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_loan_labels(df_loans):
    # Group by loan to calculate default
    loan_summary = df_loans.groupBy('loan_id', 'Customer_ID', 'loan_start_date', 'loan_amt').agg(
        F.max('overdue_amt').alias('max_overdue'),
        F.sum('overdue_amt').alias('total_overdue'),
        F.max('installment_num').alias('max_installment'),
        F.sum(F.when(F.col('overdue_amt') > 0, 1).otherwise(0)).alias('num_overdue_payments')
    )
    
    # Define default criteria
    loan_summary = loan_summary.withColumn('default_flag',
        F.when(
            (F.col('max_overdue') > F.col('loan_amt') * 0.1) |  # overdue > 10% of loan
            (F.col('num_overdue_payments') >= 3),  # 3+ overdue payments
            1
        ).otherwise(0)
    )
    
    label_store = loan_summary.select('Customer_ID', 'loan_start_date', 'default_flag')
    
    return label_store

