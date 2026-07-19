import pandas as pd
import sqlite3

conn = sqlite3.connect("traces.db")

df = pd.read_sql_query("SELECT * FROM spans", conn)

df['duration_ms'] = (df['end_time'] - df['start_time']) / 1_000_000

print("\n--- Q5: Total Duration by Span (Excluding 'rag') ---")

q5_df = df[df['name'] != 'rag'].groupby('name')['duration_ms'].sum().reset_index()
print(q5_df.to_string(index=False))

print("\n--- Q6: Token Stability Across Runs ---")

q6_df = df[df['name'] == 'llm'][['name', 'input_tokens']]
print(q6_df.to_string(index=False))

conn.close()