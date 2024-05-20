import os
import shutil
from tqdm import tqdm

def organize_gopro_dataset(dir_in, dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for folder_name in os.listdir(dir_in):
        folder_path = os.path.join(dir_in, folder_name)
        if os.path.isdir(folder_path):
            # Determine whether the folder is blurred or sharp
            if 'blurred' in folder_name.lower():
                target_folder_name = 'A'
            elif 'sharp' in folder_name.lower():
                target_folder_name = 'B'
            else:
                print(f"Skipping unrecognized folder: {folder_path}")
                continue
            
            target_folder = os.path.join(dir_out, target_folder_name)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                
            for image_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    shutil.copy(image_path, target_folder)
        else:
            print(f"Skipping non-directory: {folder_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', required=True, help='Input directory')
    parser.add_argument('--dir_out', required=True, help='Output directory')
    args = parser.parse_args()

    organize_gopro_dataset(args.dir_in, args.dir_out)