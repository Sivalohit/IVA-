#!/usr/bin/env python
# coding: utf-8

# # Task - 1

# In[13]:


import ffmpeg
import os

def fetch_video_details(path_to_video):
    try:
        probe_data = ffmpeg.probe(path_to_video)
        video_data = next(stream for stream in probe_data['streams'] if stream['codec_type'] == 'video')

        frame_rate = eval(video_data['r_frame_rate'])
        video_duration = float(video_data['duration'])
        total_frames = int(video_data['nb_frames'])
        video_width = int(video_data['width'])
        video_height = int(video_data['height'])

        show_video_details(frame_rate, video_duration, total_frames, video_width, video_height)

        return frame_rate, video_duration, total_frames, video_width, video_height
    except Exception as e:
        print(f"Error fetching video details: {e}")
        return None

def show_video_details(frame_rate, video_duration, total_frames, video_width, video_height):
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Duration: {video_duration:.2f} seconds")
    print(f"Frame Count: {total_frames} frames")
    print(f"Resolution: {video_width}x{video_height} pixels")

def save_frames(path_to_video, frames_output_dir):
    try:
        os.makedirs(frames_output_dir, exist_ok=True)
        frames_output_path = os.path.join(frames_output_dir, 'frame_%04d.png')

        ffmpeg.input(path_to_video).output(frames_output_path, start_number=0).run(capture_stdout=True, capture_stderr=True)
        print(f"Frames saved to {frames_output_dir}")
    except Exception as e:
        print(f"Error saving frames: {e}")

def main(video_file_path, frames_output_dir):

    video_info = fetch_video_details(video_file_path)

    if video_info is not None:
        save_frames(video_file_path, frames_output_dir)

if __name__ == "__main__":
    video_file_path = "C:/Users/sivalohit/OneDrive/Desktop/his game clips/27.mp4"
    frames_output_dir = "C:/Users/sivalohit/OneDrive/Desktop"

    main(video_file_path, frames_output_dir)


# # Task - 2

# In[14]:


import ffmpeg
import matplotlib.pyplot as plt

def analyze_frame_types(video_path):
    try:
        result = ffmpeg.probe(video_path, select_streams='v', show_frames=None, show_entries='frame=pict_type')
        frames = result['frames']

        frame_counts = {'I': 0, 'P': 0, 'B': 0}

        for frame in frames:
            frame_type = frame.get('pict_type')
            if frame_type in frame_counts:
                frame_counts[frame_type] += 1

        total_frames = sum(frame_counts.values())
        percentages = {frame_type: (count / total_frames * 100) if total_frames > 0 else 0 for frame_type, count in frame_counts.items()}

        display_frame_distribution(frame_counts, percentages)
        return frame_counts, percentages
    except Exception as e:
        print(f"Error analyzing frame types: {e}")
        return None, None

def display_frame_distribution(frame_counts, percentages):
    for frame_type in frame_counts:
        print(f"{frame_type}-Frames: {frame_counts[frame_type]} ({percentages[frame_type]:.2f}%)")

def visualize_frame_distribution(frame_counts):
    labels = list(frame_counts.keys())
    counts = list(frame_counts.values())

    bar_colors = ['tomato', 'steelblue', 'limegreen']
    pie_colors = ['orangered', 'dodgerblue', 'limegreen']

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color=bar_colors)
    plt.xlabel('Frame Type')
    plt.ylabel('Count')
    plt.title('Frame Type Distribution (Count)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=pie_colors)
    plt.title('Frame Type Distribution (Percentage)')
    plt.axis('equal')
    plt.show()

def main(video_path):
    frame_counts, percentages = analyze_frame_types(video_path)
    if frame_counts and percentages:
        visualize_frame_distribution(frame_counts)

if __name__ == "__main__":
    video_path = "C:/Users/sivalohit/OneDrive/Desktop/his game clips/27.mp4"
    main(video_path)


# # Task - 3

# In[9]:


import ffmpeg
import os
from PIL import Image

def extract_frames(video_path, output_folder, frame_type):

    try:
        os.makedirs(output_folder, exist_ok=True)
        output_pattern = os.path.join(output_folder, 'frame_%04d.png')
        (
            ffmpeg
            .input(video_path)
            .output(output_pattern, vf=f'select=eq(pict_type\,{frame_type})', vsync='vfr')
            .run()
        )
        print(f"{frame_type} frames extracted and saved to {output_folder}.")
    except Exception as e:
        print(f"Error extracting {frame_type} frames: {e}")

def display_extracted_frames(folder_path):

    try:
        image_files = [img for img in os.listdir(folder_path) if img.endswith('.png')]
        if not image_files:
            print(f"No frames found in {folder_path}.")
            return

        for image_file in image_files:
            img_path = os.path.join(folder_path, image_file)
            img = Image.open(img_path)
            img.show()
    except Exception as e:
        print(f"Error displaying frames from {folder_path}: {e}")

def process_video_frames(video_path, output_folders):

    for frame_type, folder in output_folders.items():
        extract_frames(video_path, folder, frame_type)
        display_extracted_frames(folder)

def main(video_path):
    output_folders = {
        'I': '/content/I_frames',
        'P': '/content/P_frames',
        'B': '/content/B_frames'
    }
    process_video_frames(video_path, output_folders)

if __name__ == "__main__":
    video_path = "C:/Users/sivalohit/OneDrive/Desktop/his game clips/27.mp4"
    main(video_path)


# # Task - 4

# In[15]:


import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
try:
    from google.colab.patches import cv2_imshow
    is_colab = True
except ImportError:
    is_colab = False

def display_frames_with_pil(frame_paths):
    if not frame_paths:
        print("No frames to display.")
        return

    plt.figure(figsize=(15, 5))
    for i, frame_path in enumerate(frame_paths):
        try:
            img = Image.open(frame_path)
            plt.subplot(1, len(frame_paths), i + 1)
            plt.imshow(img)
            plt.title(f"Frame: {os.path.basename(frame_path)}")
            plt.axis('off')
        except Exception as e:
            print(f"Error opening {frame_path} with PIL: {e}")
    plt.show()

def display_frames_with_opencv(frame_paths):
    if not frame_paths:
        print("No frames to display.")
        return

    for frame_path in frame_paths:
        try:
            img = cv2.imread(frame_path)
            if img is not None:
                if is_colab:
                    cv2_imshow(img)
                else:
                    cv2.imshow(os.path.basename(frame_path), img)
                    cv2.waitKey(0)  # Wait for a key press to close the window
                    cv2.destroyAllWindows()
            else:
                print(f"Error: Could not read {frame_path}.")
        except Exception as e:
            print(f"Error displaying {frame_path} with OpenCV: {e}")

def get_frame_paths(frame_types, base_path):
    frame_paths = []
    for frame_type in frame_types:
        frame_path = os.path.join(base_path, frame_type, "frame_0001.png")
        frame_paths.append(frame_path)
    return frame_paths

def main():
    base_path = '/content'
    frame_types = ['I_frames', 'P_frames', 'B_frames']

    frame_paths = get_frame_paths(frame_types, base_path)

    display_frames_with_pil(frame_paths)
    display_frames_with_opencv(frame_paths)

if __name__ == "__main__":
    main()


# # Task -5

# In[16]:


import os

def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        print(f"Error getting size for {file_path}: {e}")
        return 0

def compute_avg_frame_size(folder_path):
    frame_sizes = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                full_path = os.path.join(folder_path, file)
                size = get_file_size(full_path)
                if size > 0:
                    frame_sizes.append(size)

        avg_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
        return avg_size, frame_sizes
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return 0, []

def show_avg_sizes(i_frame_avg, p_frame_avg, b_frame_avg):
    print(f"Average I-Frame Size: {i_frame_avg / 1024:.2f} KB")
    print(f"Average P-Frame Size: {p_frame_avg / 1024:.2f} KB")
    print(f"Average B-Frame Size: {b_frame_avg / 1024:.2f} KB")

def main():
    i_frames_dir = '/content/I_frames'
    p_frames_dir = '/content/P_frames'
    b_frames_dir = '/content/B_frames'

    # Calculate average frame sizes
    i_frame_avg, _ = compute_avg_frame_size(i_frames_dir)
    p_frame_avg, _ = compute_avg_frame_size(p_frames_dir)
    b_frame_avg, _ = compute_avg_frame_size(b_frames_dir)

    # Display average sizes
    show_avg_sizes(i_frame_avg, p_frame_avg, b_frame_avg)

if __name__ == "__main__":
    main()


# In[ ]:




