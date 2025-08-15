import os
import shutil
import glob

# Copy models from input to working directory
input_dir = '/kaggle/input'
working_dir = '/kaggle/working'

# Find all directories in the input folder
input_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
print(f"Found {len(input_dirs)} directories in input folder")

# Look for .pth files in all input directories
model_files_found = 0
for dir_name in input_dirs:
    dir_path = os.path.join(input_dir, dir_name)
    pth_files = glob.glob(os.path.join(dir_path, "*.pth"))
    
    if pth_files:
        print(f"Found {len(pth_files)} model files in {dir_name}:")
        for file_path in pth_files:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(working_dir, file_name)
            
            # Copy the file to working directory
            try:
                shutil.copy(file_path, dest_path)
                print(f"  Copied {file_name} to working directory")
                model_files_found += 1
            except Exception as e:
                print(f"  Error copying {file_name}: {e}")

print(f"\nTotal: Copied {model_files_found} model files to working directory")
print("Working directory now contains:")
print(os.listdir(working_dir))
