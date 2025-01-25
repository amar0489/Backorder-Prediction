# For getting the file size

import os

def get_file_size(file_path):
    try:
        file_size = os.path.getsize(file_path)
        
        if file_size < 1024:
            return f"{file_size} Bytes"
        elif file_size < 1024 ** 2:
            return f"{file_size / 1024:.2f} KB"
        elif file_size < 1024 ** 3:
            return f"{file_size / (1024 ** 2):.2f} MB"
        else:
            return f"{file_size / (1024 ** 3):.2f} GB"

    except Exception as e:
        print(f"Error: {e}")
        return None

file_path = "artifacts\model.pkl.gz"
print(f"File Size: {get_file_size(file_path)}")
