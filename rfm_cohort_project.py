#Step 1:Importing libraries
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

#Step 2:Loading the dataset
df = pd.read_excel('Online Retail.xlsx')  

#Step 3:Data Cleaning
df = df.dropna(subset=['CustomerID'])            
df = df.drop_duplicates()                       
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]  

print("\n----- Sample Data -----")
print(df.head())  # Show first 5 rows

#Cohort Analysis
df['InvoiceMonth'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, 1))
df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')

def get_cohort_index(row):
    year_diff = row['InvoiceMonth'].year - row['CohortMonth'].year
    month_diff = row['InvoiceMonth'].month - row['CohortMonth'].month
    return year_diff * 12 + month_diff + 1

df['CohortIndex'] = df.apply(get_cohort_index, axis=1)

cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

# Printing first 5 rows of cohort_counts
print("\n----- Cohort Counts (first 5 rows) -----")
print(cohort_counts.head())

cohort_size = cohort_counts.iloc[:, 0]
retention = cohort_counts.divide(cohort_size, axis=0) * 100

#Printing retention summary
print("\n----- Retention Summary -----")
print(retention.round(1).head())

#Ploting Retention Heatmap
plt.figure(figsize=(12, 6))
plt.title('Customer Retention Rate (%)')
sns.heatmap(retention, annot=True, fmt=".0f", cmap="Blues")
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.show()

#Average Retention Line Chart
retention.mean(axis=0).plot(kind='line', marker='o', figsize=(10,5))
plt.title("Average Retention Over Months")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Retention Rate (%)")
plt.grid(True)
plt.show()

# -------------------- RFM Analysis --------------------
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

#RFM Quartiles
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

#Printing top 5 customers by Monetary value
print("\n----- Top 5 Customers by Monetary Value -----")
print(rfm.sort_values('Monetary', ascending=False).head())

#Showing counts of each RFM segment
rfm_counts = rfm['RFM_Score'].value_counts().sort_index()
print("\n----- RFM Segment Counts -----")
print(rfm_counts)

#Ploting RFM Segment Bar Chart
plt.figure(figsize=(14,6))
rfm_counts.plot(kind='bar', color='skyblue')
plt.title('RFM Segment Counts')
plt.xlabel('RFM Score')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()



