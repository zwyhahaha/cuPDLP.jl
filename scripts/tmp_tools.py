import pandas as pd
import os
import urllib.request

# Read the instance names from the txt file
with open('data/missing_files', 'r') as file:
    txt_instance_names = set(file.read().splitlines())

# # Read the instance names from the csv file
# csv_file_path = '/home/shanshu/opt/cuPDLP.jl/output/table/MIPLIB383/adaPDLP_time_600.0_tol_0.0001_lr_0.0.csv'
# csv_data = pd.read_csv(csv_file_path)
# csv_instance_names = set(csv_data['instance_name'])

# # Find the instance names that are in the txt file but not in the csv file
# missing_instance_names = txt_instance_names - csv_instance_names

# # Print the missing instance names
# for name in missing_instance_names:
#     print(name)

for name in txt_instance_names:
    file_url = f"https://miplib.zib.de/WebData/instances/{name}.mps.gz"
    
    output_folder = '/home/shanshu/opt/cuPDLP.jl/data/MIPLIB383'
    output_file_path = os.path.join(output_folder, f"{name}.mps.gz")

    urllib.request.urlretrieve(file_url, output_file_path)
    print(name)