import sqlite3
import pandas as pd

DB_PATH = "output/turbine_data.db"

# Connect to the database
conn = sqlite3.connect(DB_PATH)

# List tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:", tables)

# Query data
df = pd.read_sql_query("SELECT * FROM cleaned_data LIMIT 10;", conn)
df1 = pd.read_sql_query("SELECT * FROM summary_statistics LIMIT 10;", conn)
df2 = pd.read_sql_query("SELECT * FROM anomalies LIMIT 10;", conn)

print(df)
print(df1)
print(df2)

# Close connection
conn.close()
