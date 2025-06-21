'''
Que-3. Rotate an image by 45 and 90 degrees.
'''

import cv2
import numpy as np

def rotate(img, angle_deg):
    
    # Rotate the image by a given angle without cropping, adding white background
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_a = abs(rot_matrix[0, 0])
    sin_a = abs(rot_matrix[0, 1])

    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, rot_matrix, (new_w, new_h), borderValue=(255, 255, 255))

def label(img, text):
    # Add a white label bar on top of the image with text
    label_bar = np.ones((40, img.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label_bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((label_bar, img))

def resize_fixed(img, size=(400, 300)):
    
    # Resize image to exact size
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def build_display_grid(images_with_labels, grid_shape=(2, 2)):
    
    # Arrange images into a labeled grid layout
    row_imgs = []
    
    for i in range(grid_shape[0]):
        
        row = images_with_labels[i * grid_shape[1]:(i + 1) * grid_shape[1]]
        while len(row) < grid_shape[1]:
            row.append(np.full_like(row[0], 255))  # Add blank space if needed
        row_imgs.append(np.hstack(row))
    return np.vstack(row_imgs)

def main():
    
    try:
        img_path = input("Enter the image path: ").strip()
        original = cv2.imread(img_path)
        if original is None:
            raise FileNotFoundError("Unable to load image.")

        rotated_45 = rotate(original, 45)
        rotated_90 = rotate(original, 90)

        fixed_size = (400, 300)
        images = [
            label(resize_fixed(original, fixed_size), "Original"),
            label(resize_fixed(rotated_45, fixed_size), "Rotated 45°"),
            label(resize_fixed(rotated_90, fixed_size), "Rotated 90°")
        ]

        grid_img = build_display_grid(images, grid_shape=(2, 2))

        cv2.imshow("Image Rotation Comparison", grid_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as err:
        print("Error:", err)

if __name__ == "__main__":
    main()
