''' 
Que-1. To reduce the number of intensity levels in an image from 256 to 2, in integer powers
of 2. The desired number of intensity levels needs to be a variable input to your
program. '''

import cv2
import numpy as np

def validate_levels(level_count):
    
    # Ensure level count is a valid power of 2 between 2 and 256
    return level_count >= 2 and level_count <= 256 and (level_count & (level_count - 1)) == 0

def reduce_levels(grayscale_img, level_count):
    
    # Reduce grayscale intensity levels to the specified count
    factor = 256 // level_count
    return (grayscale_img // factor) * factor

def overlay_title(img, text):
    
    # Add a header with the given label above an image
    title_bar = np.full((40, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(title_bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return np.vstack((title_bar, img))

def scale_image(img, target_height=400):
    
    # Scale an image to a given height while keeping aspect ratio
    h, w = img.shape[:2]
    ratio = target_height / h
    new_dims = (int(w * ratio), target_height)
    return cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

def prepare_display_images(image_path, level_count):
    
    # Process the image and prepare display-ready original and reduced images
    if not validate_levels(level_count):
        raise ValueError("Intensity levels must be a power of 2 between 2 and 256.")

    original = cv2.imread(image_path)
    
    if original is None:
        raise FileNotFoundError("Invalid image path.")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    reduced = reduce_levels(gray, level_count).astype(np.uint8)

    # Convert for display
    gray_disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    reduced_disp = cv2.cvtColor(reduced, cv2.COLOR_GRAY2BGR)

    # Resize and label
    gray_disp = scale_image(gray_disp)
    reduced_disp = scale_image(reduced_disp)

    gray_disp = overlay_title(gray_disp, "Original Grayscale")
    reduced_disp = overlay_title(reduced_disp, f"Quantized to {level_count} Levels")

    # Match height for horizontal stack
    min_height = min(gray_disp.shape[0], reduced_disp.shape[0])
    gray_disp = cv2.resize(gray_disp, (gray_disp.shape[1], min_height))
    reduced_disp = cv2.resize(reduced_disp, (reduced_disp.shape[1], min_height))

    return np.hstack((gray_disp, reduced_disp))

def main():
    
    try:
        
        path = input("Enter the image path: ").strip()
        levels = int(input("Enter number of intensity levels (power of 2 between 2 and 256): "))
        
        result = prepare_display_images(path, levels)
        
        cv2.imshow("Intensity Level Reduction", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as error:
        print("Error:", error)

if __name__ == "__main__":
    main()
