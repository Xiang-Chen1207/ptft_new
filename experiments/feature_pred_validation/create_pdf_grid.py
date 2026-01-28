
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

def create_pdf_grid(source_dir, output_pdf):
    # Get list of images
    images = [f for f in os.listdir(source_dir) if f.endswith('.png') or f.endswith('.jpg')]
    images.sort()
    
    if not images:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(images)} images. Creating PDF...")

    # A4 size in inches (approx)
    # 3x3 grid
    
    with PdfPages(output_pdf) as pdf:
        num_images = len(images)
        images_per_page = 9
        num_pages = math.ceil(num_images / images_per_page)
        
        for page in range(num_pages):
            fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27)) # Landscape A4
            axes = axes.flatten()
            
            start_idx = page * images_per_page
            end_idx = min((page + 1) * images_per_page, num_images)
            
            page_images = images[start_idx:end_idx]
            
            for i, ax in enumerate(axes):
                if i < len(page_images):
                    img_path = os.path.join(source_dir, page_images[i])
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                        ax.set_title(page_images[i], fontsize=8)
                        ax.axis('off')
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
                        ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            print(f"Processed page {page + 1}/{num_pages}")
            
    print(f"PDF saved to {output_pdf}")

if __name__ == "__main__":
    # 1. Feat Only Scatter
    # source_dir = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_only_viz/scatter_plots"
    # output_pdf = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_only_viz.pdf"
    # create_pdf_grid(source_dir, output_pdf)
    
    # 2. Feat Full Radar
    # Note: User provided path seems to be "..._feat_full_viz\radar_charts_full", let's double check if it exists
    # If not, try "..._feat_full_viz\radar_charts"
    
    radar_full_path = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/radar_charts_full"
    # The user input path was slightly different, let's try the one that likely exists based on previous ls
    # Previous ls showed: .../radar_charts_full
    
    # But wait, user said: .../feature_metrics_eval_feat_full_viz/radar_charts_full
    # Let's check where the full model output went.
    # It likely went to feature_metrics_eval_full_viz (based on csv name feature_metrics_eval_full.csv)
    
    # Let's try to be robust.
    
    # Task 1: Full Model Radar Charts
    # Try multiple possible locations
    possible_paths_1 = [
        "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/radar_charts_full",
        "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_full_viz/radar_charts_full",
        "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_full_viz/radar_charts"
    ]
    
    dir1 = None
    for p in possible_paths_1:
        if os.path.exists(p):
            dir1 = p
            break
            
    if dir1:
        out1 = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/radar_charts_full.pdf"
        create_pdf_grid(dir1, out1)
    else:
        print(f"Directory not found for Full Model Radar Charts. Checked: {possible_paths_1}")

    # Task 2: Feat Only Radar Charts
    dir2 = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_only_viz/radar_charts"
    if os.path.exists(dir2):
        out2 = "/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_feat_only_viz_radar.pdf"
        create_pdf_grid(dir2, out2)
    else:
        print(f"Directory not found: {dir2}")
