import os
import shutil
from bing_image_downloader import downloader

def download_images():
    output_dir = 'dataset'
    target_count = 100
    
    # Define search queries
    queries = {
        "couple": "couple man and woman together",
        "solo_person_portrait": "solo person portrait"
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download images
    for label, query in queries.items():
        label_folder = os.path.join(output_dir, label)
        os.makedirs(label_folder, exist_ok=True)
        
        # Count current images in label_folder
        existing_files = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        current_count = len(existing_files)
        
        if current_count >= target_count:
            print(f"Dataset for '{label}' already has {current_count} images. Skipping.")
            continue
            
        needed = target_count - current_count
        print(f"Downloading {needed} more images for: {label} (Current: {current_count})")
        
        downloader.download(
            query,
            limit=needed,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
        
        # Merge the new folder into label_folder
        query_folder = os.path.join(output_dir, query)
        if os.path.exists(query_folder):
            new_files = os.listdir(query_folder)
            for idx, file in enumerate(new_files):
                source = os.path.join(query_folder, file)
                # Ensure unique filename to prevent collision
                new_filename = f"image_new_{current_count + idx}_{file}"
                destination = os.path.join(label_folder, new_filename)
                
                if os.path.isfile(source):
                    os.rename(source, destination)
            
            # Remove the empty query folder
            shutil.rmtree(query_folder)

if __name__ == "__main__":
    download_images()
