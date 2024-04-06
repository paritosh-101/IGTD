# import pandas as pd
# import os
# from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation

# data = pd.read_csv('D:\_workspace\IGTD_data\synthetic_easy.csv', low_memory=False, engine='c', header=0)
# data.head()

# # detach label column
# label = data['label']
# data = data.drop(['label'], axis=1)

# num_row = 5    # Number of pixel rows in image representation
# num_col = 5    # Number of pixel columns in image representation
# num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
# save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
# max_step = 30000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
# val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
#                 # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# # Select features with large variations across samples
# id = select_features_by_variation(data, variation_measure='var', num=num)
# data = data.iloc[:, id]
# # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
# norm_data = min_max_transform(data.values)
# norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# # the pixel distance ranking matrix. Save the result in Test_1 folder.

# fea_dist_method = 'Euclidean'
# image_dist_method = 'Euclidean'
# error = 'abs'
# result_dir = 'D:\_workspace\IGTD_data\Results\Test_1'
# os.makedirs(name=result_dir, exist_ok=True)
# table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
#                max_step, val_step, result_dir, error)


import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
from multiprocessing import Pool

def process_label(args):
    lbl, data_lbl, label_folder, scale, fea_dist_method, image_dist_method, save_image_size, max_step, val_step, error = args
    table_to_image(data_lbl, scale, fea_dist_method, image_dist_method, save_image_size,
                   max_step, val_step, label_folder, error)

data = pd.read_csv('/home/paritosh/workspace/IGTD_data/synthetic_easy.csv', low_memory=False, engine='c', header=0)
data.head()

# detach label column
label = data['label']
data = data.drop(['label'], axis=1)

num_row = 5    # Number of pixel rows in image representation
num_col = 5    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 30000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Select features with large variations across samples
id = select_features_by_variation(data, variation_measure='var', num=num)
data = data.iloc[:, id]
# Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.

fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '/home/paritosh/workspace/IGTD_data/Results/Test_1'
os.makedirs(name=result_dir, exist_ok=True)

# Create separate folders for each label
label_folders = {}
for lbl in label.unique():
    label_folder = os.path.join(result_dir, f"label_{lbl}")
    os.makedirs(label_folder, exist_ok=True)
    label_folders[lbl] = label_folder

# Create a pool of processes
pool = Pool()

# Prepare the arguments for each label
args_list = []
for lbl, data_lbl in norm_data.groupby(label):
    label_folder = label_folders[lbl]
    args_list.append((lbl, data_lbl, label_folder, [num_row, num_col], fea_dist_method, image_dist_method,
                      save_image_size, max_step, val_step, error))

# Run the table_to_image function in parallel for each label
pool.map(process_label, args_list)

# Close the pool
pool.close()
pool.join()