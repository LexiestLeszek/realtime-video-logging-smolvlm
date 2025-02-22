import cv2
import time
import os
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def capture_clip(cap, duration, fps, output_file):
    """
    Capture a short video clip from the webcam and save it to output_file.
    
    Parameters:
    - cap: The cv2.VideoCapture object.
    - duration: Length of the clip in seconds.
    - fps: Frames per second for the recording.
    - output_file: File path for the temporary video file.
    """
    frame_count = int(duration * fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Initialize VideoWriter - using 'mp4v' codec to write an MP4 file.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frames_captured = 0
    while frames_captured < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Camera", frame)  # Display live feed for monitoring
        frames_captured += 1
        # Check if the user pressed 'q' to exit during capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

def main():
    # Load the SmolVLM2 model and its processor.
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
    model_path
    #torch_dtype=torch.bfloat16
).to("cpu")

    
    # Initialize the webcam (typically device 0 on your MacBook)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    
    # Parameters for clip recording
    fps = 10        # frames per second
    duration = 3    # duration of each clip in seconds
    
    print("Starting realtime video logging. Press 'q' in the video window to exit.")
    
    try:
        while True:
            # Create a temporary file to save the current video clip.
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                video_filename = tmp.name
            
            # Capture a video clip from the webcam.
            capture_clip(cap, duration, fps, video_filename)
            
            # Prepare a message for the model using the temporary video file.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_filename},
                        {"type": "text", "text": "Describe this video in detail"}
                    ]
                }
            ]
            
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Generate the description of the video clip.
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Print the generated description.
            print("Detected scene description:", generated_texts[0])
            
            # Clean up the temporary video file.
            os.remove(video_filename)
            
            # Check for window closure or key press to exit.
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
