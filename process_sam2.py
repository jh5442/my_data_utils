import cv2
import numpy as np

def process_overlay_mask(original_video_path, save_mask_video_path):
    # Open input video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {original_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_mask_video_path, fourcc, fps, (width, height), isColor=False)

    # Define color range for "light blue" (in BGR)
    lower_blue = np.array([106, 115, 108])
    upper_blue = np.array([110, 155, 169])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a binary mask for pixels in the light blue range
        mask = cv2.inRange(frame, lower_blue, upper_blue)

        # Output should be single-channel (black & white)
        out.write(mask)

    cap.release()
    out.release()
    print(f"Saved binary mask video to: {save_mask_video_path}")



def process_white_mask(original_video_path, save_mask_video_path):
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {original_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_mask_video_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a binary mask: 255 where pixel is exactly white (255,255,255), else 0
        white_mask = np.all(frame == [255, 255, 255], axis=-1).astype(np.uint8) * 255

        # Write the binary mask frame
        out.write(white_mask)

    cap.release()
    out.release()
    print(f"Saved binary mask video to: {save_mask_video_path}")



if __name__ == "__main__":
    process_white_mask(original_video_path="/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_03_sam2_white.mp4",
                         save_mask_video_path="/Users/jinhuang/Desktop/work/my_data/test_03_and_04/test_03_sam2_mask_from_white.mp4")