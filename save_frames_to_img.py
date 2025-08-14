import cv2
import os


def save_video_frames(video_path, save_frame_image_folder_path):
    # Create output folder if it doesn't exist
    os.makedirs(save_frame_image_folder_path, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Construct filename with 5-digit zero-padding
        filename = f"{frame_idx:05d}.png"
        full_path = os.path.join(save_frame_image_folder_path, filename)

        # Save frame as PNG
        cv2.imwrite(full_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {save_frame_image_folder_path}")



if __name__ == '__main__':
    save_video_frames(video_path="/home/ubuntu/jin/data/test_03_and_04/test_03.mp4",
                      save_frame_image_folder_path="/home/ubuntu/jin/data/fate_zero/test_03")

    save_video_frames(video_path="/home/ubuntu/jin/data/test_03_and_04/test_04.mp4",
                      save_frame_image_folder_path="/home/ubuntu/jin/data/fate_zero/test_04")