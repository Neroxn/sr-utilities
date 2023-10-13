import os
import filecmp
import shutil

def compare_and_move_unique_files(dir1: str, dir2: str, unique_dir: str) -> int:
    """
    Compare files in two folders and move unique files from dir1 to a new directory.
    
    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        unique_dir (str): Path to the directory where unique files will be moved.
    
    Returns:
        int: Number of differing files.
    """
    differing_files = 0
    
    # Create the unique directory if it doesn't exist
    if not os.path.exists(unique_dir):
        os.mkdir(unique_dir)
    
    # List all files in both directories
    files1 = {f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
    
    # Find common files
    common_files = files1.intersection(files2)
    
    # Compare common files
    for file in common_files:
        if not filecmp.cmp(os.path.join(dir1, file), os.path.join(dir2, file)):
            differing_files += 1
    
    # Move unique files in dir1 to the unique directory
    unique_files1 = files1 - common_files
    for file in unique_files1:
        shutil.move(os.path.join(dir1, file), os.path.join(unique_dir, file))
    
    # Count unique files as differing
    differing_files += len(unique_files1) + len(files2 - common_files)
    
    return differing_files

def compare_and_remove_folders(dir1: str, dir2: str, remove_duplicates: bool = False) -> int:
    """
    Compare files in two folders and optionally remove duplicates in the first folder.
    
    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        remove_duplicates (bool, optional): Whether to remove duplicates in dir1. Defaults to False.
    
    Returns:
        int: Number of differing files.
    """
    differing_files = 0
    
    # List all files in both directories
    files1 = {f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
    
    # Find common files
    common_files = files1.intersection(files2)
    
    # Compare common files
    for file in common_files:
        if filecmp.cmp(os.path.join(dir1, file), os.path.join(dir2, file)):
            if remove_duplicates:
                os.remove(os.path.join(dir1, file))
        else:
            differing_files += 1
            
    # Count files that are only in one folder as differing
    differing_files += len(files1 - common_files) + len(files2 - common_files)
    
    return differing_files

if __name__ == "__main__":
    dir1 = "/home/borakargi/dbsr/datasets/images"
    dir2 = "/home/borakargi/dbsr/datasets/val_images"
    unique_dir = "/home/borakargi/dbsr/datasets/train_images"
    
    remove_duplicates = True  # Change this flag as needed
    # result = compare_and_remove_folders(dir1, dir2, remove_duplicates)
    result = compare_and_move_unique_files(dir1, dir2, unique_dir)

    print(f"Number of differing files: {result}")