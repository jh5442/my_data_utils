import cv2
import numpy as np



def convert_mask(original_mask_path, save_mask_path):
    """
    Invert 0 and 1 in original mask: invert foreground and background

    :param original_mask_path:
    :param save_mask_path:
    :return:
    """
    # Open the original video
    cap = cv2.VideoCapture(original_mask_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {original_mask_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create video writer
    out = cv2.VideoWriter(save_mask_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if needed
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to binary mask (just in case values aren't strictly 0/255)
        binary = (frame > 127).astype(np.uint8)

        # Invert the mask: 0 → 1, 1 → 0
        inverted = 1 - binary

        # Scale back to 0/255 to write as image
        inverted = (inverted * 255).astype(np.uint8)

        out.write(inverted)

    cap.release()
    out.release()
    print(f"Inverted mask saved to: {save_mask_path}")



if __name__ == "__main__":
    # convert_mask("/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_04_mask.mp4",
    #              "/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_04_mask_inverted.mp4")

    convert_mask("/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_03_mask_trimmed.mp4",
                 "/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_03_mask_trimmed_inverted.mp4")