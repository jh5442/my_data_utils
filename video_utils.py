import cv2
import os
from moviepy.editor import VideoFileClip
import numpy as np

test_03_original_path = "/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/mt_lab_test_videos/test_03.mp4"
test_03_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                     "mt_lab_test_videos/test_03_sora_9s_3_to_2.mp4")

test_04_original_path = "/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/mt_lab_test_videos/test_04.mp4"
test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                     "mt_lab_test_videos/test_04_sora_9s_3_to_2.mp4")

# pexel_test_video_dir = "/Users/jinhuang/Desktop/work/my_data_and_results/data/pexel_test_samples"
# pexel_processed_dir = "/Users/jinhuang/Desktop/work/my_data_and_results/data/pexel_test_samples/processed_for_runway_testing"

pexel_test_video_dir = "/Users/jinhuang/Desktop/work/my_data_and_results/data/pexel_test_samples_0919"
pexel_processed_dir = "/Users/jinhuang/Desktop/work/my_data_and_results/data/pexel_processed_for_runway_0919"


def reshape_video(reference_video_path,
                  video_path,
                  output_video_path):
    """
    Adjust the W:H ratio. Needed as post-processing for some models.

    :param reference_video_path:
    :param video_path:
    :param output_video_path:
    :return:
    """

    # Open reference video to get the desired size
    reference_video = cv2.VideoCapture(reference_video_path)
    ret, frame_a = reference_video.read()
    print("Original video:", frame_a.shape)

    if not ret:
        raise ValueError("Cannot read reference video.")
    target_size = (frame_a.shape[1], frame_a.shape[0])  # (width, height)
    reference_video.release()

    # Open video B
    video_b = cv2.VideoCapture(video_path)

    # Prepare to write the resized video B
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'XVID', 'avc1', etc.
    fps = video_b.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    # Resize each frame
    while True:
        ret, frame_b = video_b.read()
        # if not ret:
        #     break
        if not ret:
            print("End of video or error reading frame.")
            break

        if frame_b is None:
            print("Warning: Frame is None.")
            continue
        resized_frame = cv2.resize(frame_b, target_size)
        out.write(resized_frame)

    video_b.release()
    out.release()




def convert_video(input_video_path, save_video_path):
    """
    Resize the video to 1080P.

    :param input_video_path:
    :param save_video_path:
    :return:
    """
    # Load video
    clip = VideoFileClip(input_video_path)

    # Step 1: Cut to max 10 seconds
    duration = min(clip.duration, 5)
    clip = clip.subclip(0, duration)

    # Step 2: Crop to 2:3 aspect ratio (portrait)
    w, h = clip.size
    target_ratio = 2 / 3
    current_ratio = w / h

    if current_ratio > target_ratio:
        # too wide → crop width
        new_w = int(h * target_ratio)
        x_center = w // 2
        x1 = x_center - new_w // 2
        x2 = x_center + new_w // 2
        clip = clip.crop(x1=x1, y1=0, x2=x2, y2=h)
    elif current_ratio < target_ratio:
        # too tall → crop height
        new_h = int(w / target_ratio)
        y_center = h // 2
        y1 = y_center - new_h // 2
        y2 = y_center + new_h // 2
        clip = clip.crop(x1=0, y1=y1, x2=w, y2=y2)
    # if equal, no crop needed

    # Step 3: Resize (optional, if you want fixed resolution like 720x1080)
    clip = clip.resize(height=1080)  # sets height to 1080, width auto-adjusts to 720

    # Step 4: Write to file
    clip.write_videofile(save_video_path, codec="libx264", audio_codec="aac")




def adjust_FPS(video_path, save_path, target_fps=24):
    """
    Adjust the FPS of a video. Needed as post-processing for some models.

    :param video_path:
    :param save_path:
    :param target_fps:
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Read all frames from the original video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Calculate how to resample the frame sequence
    total_duration = len(frames) / original_fps
    target_frame_count = int(target_fps * total_duration)

    # Interpolate frame indices
    import numpy as np
    resampled_indices = np.linspace(0, len(frames) - 1, target_frame_count).astype(int)
    resampled_frames = [frames[i] for i in resampled_indices]

    # Write resampled frames to new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, target_fps, (frame_width, frame_height))

    for frame in resampled_frames:
        out.write(frame)
    out.release()
    print(f"Saved {len(resampled_frames)} frames at {target_fps} FPS to: {save_path}")




def paint_video_gray_based_on_mask(original_video_path,
                                   mask_video_path,
                                   save_video_path,
                                   gray_color=(128, 128, 128)):
    """
    Paint masked area in original video into gray colour.
    :param original_video_path:
    :param mask_video_path:
    :param save_video_path:
    :param gray_color:
    :return:
    """
    # Open original and mask videos
    vid = cv2.VideoCapture(original_video_path)
    mask = cv2.VideoCapture(mask_video_path)

    if not vid.isOpened() or not mask.isOpened():
        raise IOError("Failed to open original or mask video.")

    # Get video properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    while True:
        ret_vid, frame = vid.read()
        ret_mask, mask_frame = mask.read()
        if not ret_vid or not ret_mask:
            break

        # Ensure mask is single channel and binary
        if mask_frame.ndim == 3:
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_frame

        binary_mask = (mask_gray > 127).astype(np.uint8)
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)

        # Create solid gray image
        gray_img = np.full_like(frame, gray_color, dtype=np.uint8)

        # Apply: where mask == 1 → gray; where mask == 0 → original
        output_frame = np.where(mask_3ch == 1, gray_img, frame)

        out.write(output_frame.astype(np.uint8))

    vid.release()
    mask.release()
    out.release()
    print(f"Saved masked video to {save_video_path}")




def trim_video_head(original_video_path, trimmed_video_path, start_time=5):
    """
    Trimming the beginning of the video to start at start_time second.
    This was written because SAM2 had difficulty getting the full mask for the first 5 sec of MT test 03.

    :param original_video_path:
    :param trimmed_video_path:
    :param start_time:
    :return:
    """
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {original_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Compute start frame
    start_frame = int(start_time * fps)
    if start_frame >= total_frames:
        raise ValueError(f"start_time={start_time}s exceeds video length")

    # Set video to the start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Output writer
    out = cv2.VideoWriter(trimmed_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Trimmed video saved to: {trimmed_video_path}")




def trim_video_tail(video_path, trimmed_video_save_path, seconds_to_keep):
    """
    Only keeping the first seconds_to_keep second of video.
    Mainly used for testing Sora.

    :param video_path:
    :param trimmed_video_save_path:
    :param seconds_to_keep:
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Calculate how many frames to keep
    frames_to_keep = int(seconds_to_keep * fps)
    if frames_to_keep > total_frames:
        raise ValueError(f"seconds_to_keep={seconds_to_keep}s exceeds video length")

    out = cv2.VideoWriter(trimmed_video_save_path, fourcc, fps, (width, height))

    frame_idx = 0
    while frame_idx < frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Trimmed head saved to: {trimmed_video_save_path}")




def process_video(original_video_path):

    # Open the video file
    cap = cv2.VideoCapture(original_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {original_video_path}")
        return

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Total number of frames: {total_frames}")

    # Release the capture object
    cap.release()




def reshape_and_trim_video(video_folder,
                           save_video_folder):
    """

    :param video_folder:
    :param save_video_folder:
    :return:
    """
    os.makedirs(save_video_folder, exist_ok=True)

    # iterate through all files in video_folder
    for fname in os.listdir(video_folder):
        if not fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            continue

        video_path = os.path.join(video_folder, fname)
        base_name, _ = os.path.splitext(fname)

        # load video
        clip = VideoFileClip(video_path)

        # resize to 720P
        width, height = clip.size

        if width > height:
            clip_resized = clip.resize(newsize=(1080, 720))
        else:
            clip_resized = clip.resize(newsize=(720, 1080))

        # first 5 seconds
        clip_5s = clip_resized.subclip(0, min(5, clip_resized.duration))
        save_path_5s = os.path.join(save_video_folder, f"{base_name}_720p_5s.mp4")
        clip_5s.write_videofile(save_path_5s, codec="libx264", audio_codec="aac")

        # first 10 seconds
        # clip_10s = clip_resized.subclip(0, min(10, clip_resized.duration))
        # save_path_10s = os.path.join(save_video_folder, f"{base_name}_720p_10s.mp4")
        # clip_10s.write_videofile(save_path_10s, codec="libx264", audio_codec="aac")

        clip.close()
        clip_resized.close()
        clip_5s.close()
        # clip_10s.close()


if __name__ == "__main__":
    reshape_and_trim_video(video_folder=pexel_test_video_dir,
                           save_video_folder=pexel_processed_dir)