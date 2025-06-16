import json
import os
from PIL import Image, ImageDraw
import numpy as np

def load_json_data(json_path):
    """Load JSON data from file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_color_for_size(size):
    """Get a unique color for each square size"""
    colors = {
        3: (255, 0, 0),      # Red
        4: (0, 255, 0),      # Green  
        5: (0, 0, 255),      # Blue
        6: (255, 255, 0),    # Yellow
        7: (255, 0, 255),    # Magenta
        8: (0, 255, 255),    # Cyan
        9: (255, 128, 0),    # Orange
    }
    return colors.get(size, (128, 128, 128))  # Default gray

def draw_square_with_cross(draw, bottom_left_x, bottom_left_y, size, color, grid_size=45):
    """Draw a square outline with diagonal cross lines"""
    # Convert grid coordinates to pixel coordinates
    # Assuming the image corresponds to a 45x45 grid
    img_width = draw.im.size[0]
    img_height = draw.im.size[1]
    
    # Calculate pixel size per grid cell
    pixel_per_cell_x = img_width / grid_size
    pixel_per_cell_y = img_height / grid_size
    
    # Convert bottom-left grid coordinates to top-left pixel coordinates
    # Note: grid coordinates are bottom-left origin, but image coordinates are top-left origin
    pixel_x = bottom_left_x * pixel_per_cell_x
    pixel_y = img_height - (bottom_left_y + size) * pixel_per_cell_y
    
    # Calculate square dimensions in pixels
    square_width = size * pixel_per_cell_x
    square_height = size * pixel_per_cell_y
    
    # Draw square outline
    draw.rectangle([
        pixel_x, pixel_y,
        pixel_x + square_width, pixel_y + square_height
    ], outline=color, width=6)
    
    # Draw diagonal cross
    draw.line([
        pixel_x, pixel_y,
        pixel_x + square_width, pixel_y + square_height
    ], fill=color, width=6)
    
    draw.line([
        pixel_x + square_width, pixel_y,
        pixel_x, pixel_y + square_height
    ], fill=color, width=6)

def process_image(json_file_path, image_file_path, output_path):
    """Process a single JSON file and corresponding image"""
    print(f"Processing {json_file_path} with {image_file_path}")
    
    # Load JSON data
    data = load_json_data(json_file_path)
    
    # Load image
    img = Image.open(image_file_path)
    draw = ImageDraw.Draw(img)
    
    # Process each square size
    for size_str, coordinates in data.items():
        size = int(size_str)
        color = get_color_for_size(size)
        
        print(f"  Drawing {len(coordinates)} squares of size {size}")
        
        # Draw each square
        for coord in coordinates:
            bottom_left_x, bottom_left_y = coord
            draw_square_with_cross(draw, bottom_left_x, bottom_left_y, size, color)
    
    # Save the result
    img.save(output_path)
    print(f"  Saved to {output_path}")

def main():
    json_folder = "json"
    raw_images_folder = "images/transform"
    output_folder = "images/with_squares"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        # Get corresponding image file
        base_name = json_file.replace('.json', '')
        image_file = f"{base_name}.jpg"
        
        json_path = os.path.join(json_folder, json_file)
        image_path = os.path.join(raw_images_folder, image_file)
        output_path = os.path.join(output_folder, f"{base_name}.jpg")
        
        if os.path.exists(image_path):
            process_image(json_path, image_path, output_path)
        else:
            print(f"Warning: Image file {image_path} not found for {json_file}")

if __name__ == "__main__":
    main() 