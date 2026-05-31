% Program 01
% Frame Extraction Program
import cv2
import os

def extract_frames(video_path, output_folder, frames_per_second):
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video FPS: {fps}")
    
    frame_interval = fps // frames_per_second
    print(f"Frame Interval: {frame_interval}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % frame_interval == 0:

            output_file = os.path.join(output_folder, f"vid_{frame_count // frame_interval}.jpg")
            print(f"Saving frame {frame_count // frame_interval} to {output_file}")
            
            cv2.imwrite(output_file, frame)
        
        frame_count += 1
    
    cap.release()
    print("Frame extraction completed.")

video_path = "..."
output_folder = "..."
frames_per_second = 2

os.makedirs(output_folder, exist_ok=True)
extract_frames(video_path, output_folder, frames_per_second)
