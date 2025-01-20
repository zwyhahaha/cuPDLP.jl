import pandas as pd
import numpy as np

def shifted_geometric_mean(data, shift=1):
    shifted_data = data + shift
    return np.exp(np.mean(np.log(shifted_data)))

def compute_shifted_geometric_mean(df, selected_columns, shift=1):
    selected_data = df[selected_columns]
    return selected_data.apply(lambda x: shifted_geometric_mean(x, shift))

def compute_mean(df, selected_columns):
    return df[selected_columns].mean()

# file_path = 'output/table/MIPLIB/xxhyper_merged_time_60.0_tol_0.0001.csv'
file_path = 'output/table/netlib/xxhyper_merged_time_180.0_tol_0.0001.csv'
# file_path = 'output/table/netlib/nonxhyper_merged_time_180.0_tol_0.0001.csv'
# file_path = 'output/table/MIPLIB2017/nonhyper_merged_time_600.0_tol_0.0001.csv'

iteration_cols = ['iteration_count_adaPDLP', 'iteration_count_osPDLP', 'iteration_count_ifosPDLP']
solve_time_cols = ['solve_time_sec_adaPDLP', 'solve_time_sec_osPDLP', 'solve_time_sec_ifosPDLP']
shift = 10

df = pd.read_csv(file_path)

termination_columns = [col for col in df.columns if col.startswith('termination_string')]
df_filtered = df[df[termination_columns].apply(lambda x: all(x == 'OPTIMAL'), axis=1)]

iter_result = compute_shifted_geometric_mean(df_filtered, iteration_cols, shift)
print("Iterations SGM", shift)
print(iter_result)

iter_result = compute_mean(df_filtered, iteration_cols)
print("Iterations Mean")
print(iter_result)

time_result = compute_shifted_geometric_mean(df_filtered, solve_time_cols, 0)
print("Solving Time SGM", 0)
print(time_result)

time_result = compute_mean(df_filtered, solve_time_cols)
print("Solving Time Mean")
print(time_result)

total_length = len(df_filtered)
count_less_equal = (df_filtered['iteration_count_osPDLP'] < df_filtered['iteration_count_adaPDLP']).sum()
count_equal = (df_filtered['iteration_count_osPDLP'] == df_filtered['iteration_count_adaPDLP']).sum()
count_greater_equal = (df_filtered['iteration_count_osPDLP'] > df_filtered['iteration_count_adaPDLP']).sum()

print("Count of iteration_count_osPDLP <= iteration_count_adaPDLP:", count_less_equal, "out of", total_length)
print("Count of iteration_count_osPDLP == iteration_count_adaPDLP:", count_equal, "out of", total_length)
print("Count of iteration_count_osPDLP >= iteration_count_adaPDLP:", count_greater_equal, "out of", total_length)

count_less_equal = (df_filtered['iteration_count_ifosPDLP'] < df_filtered['iteration_count_adaPDLP']).sum()
count_equal = (df_filtered['iteration_count_ifosPDLP'] == df_filtered['iteration_count_adaPDLP']).sum()
count_greater_equal = (df_filtered['iteration_count_ifosPDLP'] > df_filtered['iteration_count_adaPDLP']).sum()

print("Count of iteration_count_ifosPDLP <= iteration_count_adaPDLP:", count_less_equal, "out of", total_length)
print("Count of iteration_count_ifosPDLP == iteration_count_adaPDLP:", count_equal, "out of", total_length)
print("Count of iteration_count_ifosPDLP >= iteration_count_adaPDLP:", count_greater_equal, "out of", total_length)