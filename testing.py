import pandas as pd


col_dict = {
    'queue_type': 0,
    'burst_allowance': 1,
    'drop_probability': 2,
    'current_queue_delay': 3,
    'accumulated_probability': 4,
    'length_in_bytes': 5,
    'packet_length': 6
}

df = pd.read_csv("exp_pool_data.csv")

print(df.columns)

# df = df.iloc[:5600]

print("Entire Dataset")
print("No. Datapoints:",df.shape)
print("median current_queue_delay",df['state_3'].median())
print("mean current_queue_delay",df['state_3'].mean())

print("median reward",df['rewards'].median())
print("mean reward",df['rewards'].mean())
print("describe reward",df['rewards'].describe())
