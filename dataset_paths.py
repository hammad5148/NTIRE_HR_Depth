import os
from glob import glob

if __name__ == '__main__':
    # Using your full Windows paths
    dataset_dir = "C./dataset/test_mono_nogt"
    output_path = "./dataset_paths/test.txt"
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        # Loop through category folders (e.g., Desk, Sanitaries)
        for cat_name in os.listdir(dataset_dir):
            cat_path = os.path.join(dataset_dir, cat_name)
            if not os.path.isdir(cat_path): continue
            for cam_id in ["00", "02"]:
                cam_folder = os.path.join(cat_path, f"camera_{cam_id}")
                
                # Check ONLY if the camera folder exists
                if os.path.exists(cam_folder):
                    images = glob(os.path.join(cam_folder, "*.png"))
                    for img_path in images:
                        # Get the absolute (whole) path
                        full_path = os.path.abspath(img_path)
                        
                        # Write the full path to the text file
                        f.write(f"{full_path}\n")

    print(f"File saved successfully at: {output_path}")