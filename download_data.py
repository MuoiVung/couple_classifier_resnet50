import os
from bing_image_downloader import downloader

def download_images():
    output_dir = 'dataset'
    
    # Define search queries
    queries = {
        "couple": "couple man and woman together",
        "single": "single person solo photoshoot"
    }
    
    # Download images
    for label, query in queries.items():
        print(f"Downloading images for: {label}")
        downloader.download(
            query,
            limit=100,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
        
        # Rename the folder to the label if it's different
        query_folder = os.path.join(output_dir, query)
        label_folder = os.path.join(output_dir, label)
        
        if os.path.exists(query_folder):
            if os.path.exists(label_folder):
                # Move files if label_folder already exists
                for file in os.listdir(query_folder):
                    os.rename(os.path.join(query_folder, file), os.path.join(label_folder, file))
                os.rmdir(query_folder)
            else:
                os.rename(query_folder, label_folder)

if __name__ == "__main__":
    download_images()
