import cv2
import time
import os
import tempfile
import threading
import queue
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class VideoCaptureThread(threading.Thread):
    """
    Continuously grab frames from the webcam in a separate thread.
    """
    def __init__(self, src=0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open video source.")
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # Sleep a little if no frame is read to avoid busy waiting
                time.sleep(0.01)

    def read(self):
        with self.lock:
            # Return a copy of the current frame if available
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.join()
        self.cap.release()

def capture_clip(capture_thread, duration, fps, output_file):
    """
    Capture a short video clip using frames from the capture thread.
    """
    frame_interval = 1.0 / fps
    end_time = time.time() + duration
    # Get frame dimensions from the capture thread's VideoCapture object
    width = int(capture_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while time.time() < end_time:
        frame = capture_thread.read()
        if frame is not None:
            out.write(frame)
            cv2.imshow("Camera", frame)
        # Sleep to roughly match the desired fps
        time.sleep(frame_interval)
        # Allow exit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

def processing_worker(clip_queue, processor, model, device):
    """
    Worker thread that processes video clips by generating a description.
    """
    while True:
        video_filename = clip_queue.get()
        if video_filename is None:
            break  # Termination signal received
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
        ).to(device)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("Detected scene description:", generated_texts[0])
        os.remove(video_filename)
        clip_queue.task_done()

def main():
    # Load the model and its processor once outside the loop.
    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    device = "cpu"
    model = AutoModelForImageTextToText.from_pretrained(model_path).to(device)

    try:
        capture_thread = VideoCaptureThread(src=0)
    except RuntimeError as e:
        print(str(e))
        return
    capture_thread.start()

    # Parameters for clip recording
    fps = 10         # frames per second
    duration = 3     # duration (seconds) for each clip

    # Queue to hand off captured clip filenames for processing
    clip_queue = queue.Queue()

    # Start the processing worker thread
    proc_thread = threading.Thread(target=processing_worker, args=(clip_queue, processor, model, device))
    proc_thread.daemon = True
    proc_thread.start()

    print("Starting realtime video logging. Press 'q' in the video window to exit.")

    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                video_filename = tmp.name

            capture_clip(capture_thread, duration, fps, video_filename)
            clip_queue.put(video_filename)
            # Exit if the display window has been closed
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        capture_thread.stop()
        clip_queue.put(None)  # Signal the processing thread to terminate
        proc_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
