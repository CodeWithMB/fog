from django.db import models

class DetectedFace(models.Model):
    face_encoding = models.BinaryField()  # Store face encoding
    timestamp = models.DateTimeField(auto_now_add=True)  # Timestamp for when detected
    image = models.ImageField(upload_to="detected_faces/", null=True, blank=True)  # Store face image (Optional)

    def __str__(self):
        return f"Face detected on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
