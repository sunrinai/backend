import os

import cv2
import numpy as np
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("./clipvitbasepatch32")
processor = CLIPProcessor.from_pretrained("./clipvitbasepatch32")

# OpenAI API key setup

client = OpenAI(
    api_key="",  # This is the default and can be omitted
)

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (fps * interval) == 0:
            frames.append((count // fps, frame))
        count += 1

    cap.release()
    return frames


def process_transcript(transcript):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
            {
                "role": "user",
                "content": f"Summarize key points and extract important keywords(format: 'keypoints', 'keypoints', ... , 'keypoints'): {transcript}",
            }
        ]
    )
    print(response)
    return response.choices[0].message.content.strip().split(", ")


def rank_frames_with_clip(frames, keywords):
    scores = []
    for timestamp, frame in frames:
        # Convert frame to CLIP input format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(text=keywords, images=frame_rgb, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        # Aggregate scores (e.g., max score across all text queries)
        score = outputs.logits_per_image.max().item()  # Or use mean() if needed
        scores.append((timestamp, frame, score))

    return sorted(scores, key=lambda x: x[2], reverse=True)

def is_duplicate(frame, processed_hashes, threshold=5):
    """
    Check if the given frame is a duplicate based on perceptual hashing.
    """
    # Convert the frame to PIL Image format
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Compute the perceptual hash
    current_hash = imagehash.average_hash(pil_image)

    # Compare with existing hashes
    for h in processed_hashes:
        if abs(current_hash - h) <= threshold:  # Adjust threshold for similarity tolerance
            return True

    # Add the current hash to the processed list
    processed_hashes.append(current_hash)
    return False

def save_screenshots(ranked_frames, output_dir="screenshots", top_n=5, score_threshold=0.05):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_saved_score = None
    saved_count = 0

    paths = []

    for i, (timestamp, frame, score) in enumerate(ranked_frames):
        if saved_count >= top_n:
            break

        # Skip frames with very close scores
        if last_saved_score is not None and abs(score - last_saved_score) < score_threshold:
            print(f"Skipping frame at timestamp {timestamp} due to close score: {score}")
            continue

        # Save screenshot
        screenshot_path = f"{output_dir}/screenshot_{saved_count + 1}_timestamp_{timestamp}.png"
        cv2.imwrite(screenshot_path, frame)
        paths.append(screenshot_path)
        print(f"Saved screenshot: {screenshot_path} | Score: {score}")

        last_saved_score = score
        saved_count += 1

    return paths

def is_shared_screen(frame):
    """
    Determine if the current frame contains a shared screen.
    Uses heuristics such as edge density and absence of multiple faces.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces or a single face occupies a small portion of the screen, check for screen share features
    if len(faces) <= 1:
        # Check for edge density (shared screens often have sharp edges and text)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1])
        if edge_density > 0.05:  # Threshold for shared screen detection
            return True

    return False


def process_video(id, video_path, transcript, interval=2):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    frames_to_process = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (fps * interval) == 0:
            if is_shared_screen(frame):
                frames_to_process.append((count // fps, frame))
        count += 1

    cap.release()

    if not frames_to_process:
        print("No shared screens detected in the video.")
        return

    # Process frames with CLIP and save screenshots
    keywords = process_transcript(transcript)
    ranked_frames = rank_frames_with_clip(frames_to_process, keywords)
    return {"images": save_screenshots(ranked_frames, f"jobs/{id}/"), "keypoints": keywords}
