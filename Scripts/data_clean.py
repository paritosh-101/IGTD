import os

# Specify the path to the root folder
root_folder = "/home/paritosh/workspace/IGTD_data/Results/Test_1"

# Specify the image file extensions you want to keep
image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

# Traverse through the class folders inside the root folder
for foldername, subfolders, filenames in os.walk(root_folder):
    # Check if the current folder is a class folder (not the root folder itself)
    if foldername != root_folder:
        # Iterate over all the files in the class folder
        for filename in filenames:
            # Check if the file extension is not in the list of image extensions
            if os.path.splitext(filename)[1].lower() not in image_extensions:
                # Delete the file
                file_path = os.path.join(foldername, filename)
                os.remove(file_path)
                # print(f"Deleted: {file_path}")


