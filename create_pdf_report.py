import os
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

def create_pdf(input_dir, output_pdf):
    # Get all PNG files
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Creating PDF...")

    # Layout settings
    rows = 3
    cols = 3
    imgs_per_page = rows * cols
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        num_pages = math.ceil(len(image_files) / imgs_per_page)
        
        for page in range(num_pages):
            print(f"Processing page {page + 1}/{num_pages}...")
            
            # Create a figure for the page
            # A4 size is roughly 8.27 x 11.69 inches. Let's make it square-ish or portrait
            fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
            axes = axes.flatten() # Flatten 2D array of axes to 1D
            
            # Get images for this page
            start_idx = page * imgs_per_page
            end_idx = min(start_idx + imgs_per_page, len(image_files))
            page_images = image_files[start_idx:end_idx]
            
            for i, ax in enumerate(axes):
                if i < len(page_images):
                    img_path = page_images[i]
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                        ax.axis('off') # Hide axes
                        
                        # Optional: Add title based on filename
                        # basename = os.path.basename(img_path).replace('.png', '').replace('_', ' ')
                        # ax.set_title(basename, fontsize=8)
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
                        ax.axis('off')
                else:
                    # Hide unused subplots
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"PDF saved to {output_pdf}")

if __name__ == "__main__":
    input_dir = "/vePFS-0x0d/home/cx/ptft/feature_scatter_plots"
    output_pdf = "/vePFS-0x0d/home/cx/ptft/feature_scatter_plots_report.pdf"
    create_pdf(input_dir, output_pdf)
