'''
Que-2. Load an image and then perform a simple spatial 3x3 average of image pixels. Repeat
the process for a 10x10 neighborhood and again for a 20x20 neighborhood.
'''

import cv2
import numpy as np

def mean_filter(image, size):
    
    # Apply a mean filter of specified kernel size
    return cv2.blur(image, (size, size))

def label_image(image, title):
    
    # Add a white label bar with title above an image
    bar_height = 40
    label_bar = np.ones((bar_height, image.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((label_bar, image))

def resize_image(img, height=300):
    
    # Resize image to a fixed height while keeping aspect ratio
    h, w = img.shape[:2]
    scale_ratio = height / h
    return cv2.resize(img, (int(w * scale_ratio), height), interpolation=cv2.INTER_AREA)

def process_filters(image_path):
    
    # Process the grayscale image with multiple mean filters and return a 2x2 grid
    original = cv2.imread(image_path)
    
    if original is None:
        raise ValueError("Invalid image path or unable to load image.")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply different mean filter sizes
    filtered_images = {
        "Original Grayscale": gray,
        "3x3 Mean Filter": mean_filter(gray, 3),
        "10x10 Mean Filter": mean_filter(gray, 10),
        "20x20 Mean Filter": mean_filter(gray, 20)
    }

    # Convert to BGR and prepare labeled images
    labeled_blocks = []
    
    for label, img in filtered_images.items():
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = resize_image(bgr)
        labeled = label_image(resized, label)
        labeled_blocks.append(labeled)

    # Combine into a 2x2 grid
    top = np.hstack((labeled_blocks[0], labeled_blocks[1]))
    bottom = np.hstack((labeled_blocks[2], labeled_blocks[3]))
    return np.vstack((top, bottom))

def main():
    
    try:
        img_path = input("Enter the path to the image: ").strip()
        result_grid = process_filters(img_path)

        cv2.imshow("Mean Filtering Comparison", result_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # ✅ Properly ends the try block

    except Exception as err:
        print("Error:", err)

if __name__ == "__main__":
    main()
