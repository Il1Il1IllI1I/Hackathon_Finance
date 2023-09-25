import pandas as pd

# Load both DataFrames
s_df = pd.read_csv('date_last_correct.csv')
formatted_kospi_corrected_df = pd.read_csv('return_kospi_correct.csv')

# Check the length of both DataFrames to ensure they are the same
if len(s_df) != len(formatted_kospi_corrected_df):
    raise ValueError("The length of the two DataFrames is not equal. Cannot proceed with the merge.")

# Create a new DataFrame, combining columns from both DataFrames by order.
investing_df = pd.DataFrame()
investing_df['base_date'] = s_df['base_date']
investing_df['before_last'] = s_df['before_last']
investing_df['m_return'] = formatted_kospi_corrected_df['m_return']
investing_df['kospi'] = formatted_kospi_corrected_df['kospi']

# Save the combined DataFrame as 'investing.csv'
investing_df.to_csv('investing.csv', index=False)
investing_df.head()
