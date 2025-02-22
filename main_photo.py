import os
import cv2
import time
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def capture_frame(cap):
    """Capture a single frame from the webcam and return it."""
    ret, frame = cap.read()
    if not ret:
        return None
    cv2.imshow("Camera", frame)  # Display live feed
    return frame

def main():
    # Initialize model and processor
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path).to("cpu")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Starting realtime image logging. Press 'q' to exit.")
    last_process_time = time.time()
    
    try:
        while True:
            # Capture frame every 1 second
            if time.time() - last_process_time >= 1.0:
                frame = capture_frame(cap)
                if frame is None:
                    continue
                
                # Save frame to temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    image_path = tmp.name

                # Prepare model input
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_path},
                        {"type": "text", "text": "Describe what you see in this image"}
                    ]
                }]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                # Generate description
                generated_ids = model.generate(**inputs, max_new_tokens=64)
                description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"[{time.strftime('%H:%M:%S')}] Description:", description)
                
                # Cleanup
                os.remove(image_path)
                last_process_time = time.time()

            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
