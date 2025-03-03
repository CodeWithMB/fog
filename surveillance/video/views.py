from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
import cv2
import numpy as np
import pickle
import face_recognition
from ultralytics import YOLO
from .models import DetectedFace
import requests
import os
from django.conf import settings


# Load YOLO model
model = YOLO("yolov8n.pt")

# Global Variables
known_faces = []  # Store known face encodings
tracked_people = set()  # Store unique person IDs
new_face_detected = False
frame_counter = 0  # Optimize frame processing
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_alert(message):
    """Sends an intrusion alert via Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)
    

def save_detected_face(face_image, face_id):
    media_faces_path = os.path.join(settings.MEDIA_ROOT, "faces")
    
    # Debug: Check if MEDIA_ROOT is correct
    print("MEDIA_ROOT:", settings.MEDIA_ROOT)
    print("Full face path:", media_faces_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(media_faces_path):
        try:
            os.makedirs(media_faces_path)
            print("Created directory:", media_faces_path)
        except Exception as e:
            print("Error creating directory:", e)

    # Save the image
    face_path = os.path.join(media_faces_path, f"face_{face_id}.jpg")
    try:
        cv2.imwrite(face_path, face_image)
        print("Saved face image at:", face_path)
    except Exception as e:
        print("Error saving image:", e)
    
    return f"faces/face_{face_id}.jpg"


def detect_intrusion(frame):
    """Detects new faces, recognizes existing ones, and stores in DB."""
    global new_face_detected
    global frame_counter

    if frame_counter % 5 != 0:  # Process face recognition every 5 frames (optimization)
        return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encodings[i], tolerance=0.6)

        if not any(matches):  # If new face detected
            new_face_detected = True
            print("🚨 New Person Detected!")
            known_faces.append(face_encodings[i])

            # Store in database
            encoded_data = pickle.dumps(face_encodings[i])
            detected_face = DetectedFace.objects.create(face_encoding=encoded_data)

            # Save detected face image
            face_image = rgb_frame[top:bottom, left:right]  # Crop face
            face_image_path = f"media/detected_faces/face_{detected_face.id}.jpg"
            cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))  # Save as image
            detected_face.image = face_image_path
            detected_face.save()

            # Send alert
            send_telegram_alert("🚨 New Person Detected in Surveillance System!")

            # Draw Red Bounding Box for New Face
            label = "New Person"
            color = (0, 0, 255)  # Red for new faces
        else:
            label = f"Person {matches.index(True) + 1}" if True in matches else "Unknown"
            color = (0, 255, 0)  # Green for recognized faces

        # Draw Bounding Box & Label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def generate_frames(mode):
    """Processes video frames based on the selected mode."""
    cap = cv2.VideoCapture(0)  # Open webcam
    global tracked_people
    global frame_counter

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1  # Increment frame counter

        # Run YOLO detection
        results = model.track(frame, persist=True)

        people_count = 0
        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls[0])] == "person":
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Track unique IDs for people counting
                    if box.id is not None:
                        tracked_people.add(int(box.id[0]))  # Add to unique people list

                    # Draw Bounding Box & Label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame, f'Person {people_count}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Feature 1: Occupancy Detection
        if mode == "occupancy":
            cv2.putText(frame, f'Occupancy: {people_count}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Feature 2: Boundary Crossing Detection
        elif mode == "boundary":
            line_y = 250  # Set boundary line position
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)  # Red boundary line
            for r in results:
                for box in r.boxes:
                    if r.names[int(box.cls[0])] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_center_y = (y1 + y2) // 2  # Midpoint of person
                        if person_center_y > line_y:
                            cv2.putText(frame, "🚨 Boundary Crossed!", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Feature 3: Unique People Counter
        elif mode == "people_count":
            cv2.putText(frame, f'Unique Visitors: {len(tracked_people)}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Feature 4: Face Recognition & Intrusion Detection
        frame = detect_intrusion(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def video_feed(request, mode):
    """Streams processed video based on selected mode."""
    return StreamingHttpResponse(generate_frames(mode),
                                 content_type="multipart/x-mixed-replace; boundary=frame")

def check_new_faces(request):
    """API to check if a new face was detected."""
    global new_face_detected
    response = {'new_face_detected': new_face_detected}
    new_face_detected = False  # Reset flag
    return JsonResponse(response)

def home(request):
    """Renders home page with detected faces."""
    faces = DetectedFace.objects.all()
    return render(request, "video/home.html", {"faces": faces})
