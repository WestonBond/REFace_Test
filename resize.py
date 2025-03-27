import os
from PIL import Image

def main():
    input_dir = "dataset/FaceData/FFHQ/images1024"
    output_dir = "dataset/FaceData/FFHQ/images512"
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(input_path) as img:
                img_resized = img.resize((512, 512), Image.LANCZOS)
                img_resized.save(output_path, 'PNG', quality=100)
                print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main() 