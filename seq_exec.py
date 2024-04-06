import subprocess
import sys

# List of argument sets for each run
argument_sets = [
    {
        "data": "/home/paritosh/workspace/IGTD_data/Results/Test_1",
        "dataset-name": "custom_grayscale",
        "crop-size": 224,
        "arch": "resnet18",
        "epochs": 1,
        "b": 128
    },
    {
        "data": "/home/paritosh/workspace/IGTD_data/Results/Test_1",
        "dataset-name": "custom_grayscale",
        "crop-size": 224,
        "arch": "resnet34",
        "epochs": 1,
        "b": 128
    },
    {
        "data": "/home/paritosh/workspace/IGTD_data/Results/Test_1",
        "dataset-name": "custom_grayscale",
        "crop-size": 224,
        "arch": "resnet50",
        "epochs": 1,
        "b": 128
    }
    # Add more argument sets as needed
]

# Iterate over the argument sets and execute the script for each one
for args in argument_sets:
    command = [
        "python",
        "custom_SimCLR/run.py",
        "-data", args["data"],
        "-dataset-name", args["dataset-name"],
        "-crop-size", str(args["crop-size"]),
        "--arch", args["arch"],
        "--epochs", str(args["epochs"]),
        "-b", str(args["b"])
    ]
    
    try:
        # Execute the script using subprocess
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing script with arguments: {args}")
        print(f"Error message: {str(e)}")
        sys.exit(1)  # Exit with an error code if any execution fails

print("All executions completed successfully.")