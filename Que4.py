'''
Que-4. For every 3x3 block of the image (without overlapping), replace all the corresponding
9 pixels by their average. This operation simulates reducing the image spatial
resolution. Repeat this for 5x5 blocks and 7x7 blocks.
'''

import cv2
import numpy as np

def average_pooling(img, ksize):
    
    
    # Reduce image resolution by replacing non-overlapping blocks with their average.
    # Works on grayscale images.
    
    height, width = img.shape
    h_trim = height - (height % ksize)
    w_trim = width - (width % ksize)
    cropped = img[:h_trim, :w_trim].copy()

    for i in range(0, h_trim, ksize):
        for j in range(0, w_trim, ksize):
            block = cropped[i:i+ksize, j:j+ksize]
            avg = int(np.mean(block))
            cropped[i:i+ksize, j:j+ksize] = avg

    return cropped

def add_caption(img, caption):
    
    # Overlay a text label above the image
    caption_bar = np.ones((40, img.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(caption_bar, caption, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return np.vstack((caption_bar, img))

def resize_uniform(img, size=(400, 300)):
    
    # Resize image to a uniform fixed size (width x height
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def process_and_prepare(image_path):
    
    # Process the image and prepare labeled versions with reduced resolutions
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Unable to load image from the specified path.")

    # Apply average pooling for multiple block sizes
    versions = {
        "Original Grayscale": img,
        "Block Avg 3x3": average_pooling(img, 3),
        "Block Avg 5x5": average_pooling(img, 5),
        "Block Avg 7x7": average_pooling(img, 7),
    }

    fixed_size = (400, 300)
    labeled_images = []

    for label, image in versions.items():
        bgr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized = resize_uniform(bgr_img, fixed_size)
        labeled = add_caption(resized, label)
        labeled_images.append(labeled)

    return labeled_images

def arrange_grid(images, rows=2, cols=2):
    
    # Arrange labeled images into a grid of given rows and columns
    grid_rows = []
    for i in range(rows):
        row = images[i*cols:(i+1)*cols]
        if len(row) < cols:
            row += [np.full_like(row[0], 255)] * (cols - len(row))
        grid_rows.append(np.hstack(row))
    return np.vstack(grid_rows)

def main():
    
    try:
        path = input("Enter image path: ").strip()
        images = process_and_prepare(path)
        final_grid = arrange_grid(images, rows=2, cols=2)

        cv2.imshow("Spatial Resolution Reduction", final_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as err:
        print("Error:", err)

if __name__ == "__main__":
    main()
