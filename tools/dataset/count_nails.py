import os
import argparse

def count_nails_in_folder(folder_path):
    total_nails = 0
    invalid_subfolders = []
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            try:
                nails = int(item.split('(')[1].split(' ')[0])
                total_nails += nails
            except (IndexError, ValueError):
                invalid_subfolders.append(item)
    
    if invalid_subfolders:
        print(f"Invalid subfolders in '{os.path.basename(folder_path)}': {', '.join(invalid_subfolders)}")
    
    return total_nails

def main(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    total_nails = 0
    for root, dirs, _ in os.walk(folder_path):
        if root == folder_path:  # Only search in subfolders
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                nails_count = count_nails_in_folder(dir_path)
                print(f"Folder '{dir_name}' contains {nails_count} nails.")
                total_nails += nails_count

    print(f"\nTotal nails in all folders: {total_nails}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count nails in subfolders based on subfolder names.")
    parser.add_argument("folder_path", type=str, help="Path to the folder to search recursively (depth 1).")
    args = parser.parse_args()

    main(args.folder_path)