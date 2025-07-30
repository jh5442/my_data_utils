import cv2
import os
from moviepy.editor import VideoFileClip

# test_03_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
#                 "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03.mp4")
# test_03_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
#                 "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03_removal.mp4")
# test_03_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
#                      "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03_removal_reshaped.mp4")

# test_03_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/"
#                 "results/gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_03_removal/test_03.mp4")
# test_03_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_wan2.1_1.3B/test_03_removal.mp4")
# test_03_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                      "gen_omnimatte_wan2.1_1.3B/test_03_removal_reshaped.mp4")

# test_04_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
#                          "mt_lab_test_videos/test_04.mp4")
# test_04_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_04_removal/casper_outputs/gradio_demo-2dcdf102-fg=-1-0001.mp4")
# test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_04_removal/result_reshaped.mp4")

# test_04_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
#                          "mt_lab_test_videos/test_04.mp4")
# test_04_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_wan2.1_1.3B/test_04_removal.mp4")
# test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_wan2.1_1.3B/test_04_removal_reshaped.mp4")
#
# test_06_original_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
#                          "mt_lab_test_videos/test_06.mp4")
# test_06_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_06_removal/casper_outputs/gradio_demo-560f6581-fg=-1-0001.mp4")
# test_06_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/results/"
#                 "gen_omnimatte_CogVideoX-Fun-V1.5-5b-InP/test_06_removal/casper_outputs/result_reshaped.mp4")

test_03_original_path = "/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/mt_lab_test_videos/test_03.mp4"
test_03_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                     "mt_lab_test_videos/test_03_sora_9s_3_to_2.mp4")

test_04_original_path = "/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/mt_lab_test_videos/test_04.mp4"
test_04_save_path = ("/Users/jinhuang/Desktop/work/my_data/demo_for_design_and_product/data/"
                     "mt_lab_test_videos/test_04_sora_9s_3_to_2.mp4")


def reshape_video(reference_video_path,
                  video_path,
                  output_video_path):

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




def trim_video(original_video_path, trimmed_video_path, start_time=5):
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




if __name__ == "__main__":
    adjust_FPS(video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_03_removed_8s.mp4",
               save_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_03_removed_8s_fps_24.mp4")

    adjust_FPS(video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_04_removed_5s.mp4",
               save_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_04_removed_5s_fps_24.mp4")

    # convert_video(input_video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_03_removed.mp4",
    #               save_video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_03_removed_8s.mp4")

    # convert_video(input_video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_04_removed.mp4",
    #               save_video_path="/Users/jinhuang/Desktop/work/my_data/results_collection/test_04_removed_5s.mp4")


    # convert_video(input_video_path=test_04_original_path,
    #               save_video_path=test_04_save_path)

    # reshape_video(reference_video_path=test_03_original_path,
    #               video_path=test_03_path,
    #               output_video_path=test_03_save_path)

    # reshape_video(reference_video_path=test_04_original_path,
    #               video_path=test_04_path,
    #               output_video_path=test_04_save_path)

    # reshape_video(reference_video_path=test_06_original_path,
    #               video_path=test_06_path,
    #               output_video_path=test_06_save_path)