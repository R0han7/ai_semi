import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
import os
from pathlib import Path
import glob

# Try to import from the repository's modules
try:
    from backbones import get_model
except ImportError:
    print("backbones module not found. Make sure you're in the correct directory.")
    
# Alternative face alignment using OpenCV
def simple_face_align(face_input, target_size=(112, 112)):
    """
    Simple face alignment for face input (can be image path or numpy array)
    """
    if isinstance(face_input, str):
        # If it's a file path
        img = cv2.imread(face_input)
        if img is None:
            raise ValueError(f"Could not load image from {face_input}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect face in the image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Take the largest face
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            padding = int(0.2 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            
            face_crop = img_rgb[y_start:y_end, x_start:x_end]
        else:
            # If no face detected, use the whole image
            face_crop = img_rgb
    else:
        # If it's already a numpy array (from webcam)
        if len(face_input.shape) == 3 and face_input.shape[2] == 3:
            face_crop = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        else:
            face_crop = face_input
    
    # Convert to PIL and resize
    face_pil = Image.fromarray(face_crop)
    return face_pil.resize(target_size)

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

class RealTimeFaceRecognition:
    def __init__(self, model_name="edgeface_s_gamma_05", images_folder="images", similarity_threshold=0.6):
        self.model_name = model_name
        self.images_folder = images_folder
        self.similarity_threshold = similarity_threshold
        
        # Create images folder if it doesn't exist
        os.makedirs(images_folder, exist_ok=True)
        
        # Load model
        self.model = get_model(model_name)
        checkpoint_path = f'checkpoints/{model_name}.pt'
        
        # Load pre-trained weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load reference embeddings from images folder
        self.reference_embeddings = {}
        self.load_reference_images()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
            
        # Set camera properties (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Face recognition system initialized with {len(self.reference_embeddings)} reference faces.")
        print("Press 'q' to quit, 'r' to reload reference images.")
    
    def load_reference_images(self):
        """Load all images from the images folder and extract embeddings"""
        print(f"Loading reference images from '{self.images_folder}' folder...")
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext.upper())))
        
        if not image_files:
            print(f"No images found in '{self.images_folder}' folder!")
            print("Please add some face images to compare against.")
            return
        
        self.reference_embeddings = {}
        
        for image_path in image_files:
            try:
                # Extract person name from filename (without extension)
                person_name = Path(image_path).stem
                
                # Extract embedding from the reference image
                embedding = self.extract_face_embedding_from_path(image_path)
                
                if embedding is not None:
                    self.reference_embeddings[person_name] = embedding
                    print(f"Loaded reference for: {person_name}")
                else:
                    print(f"Could not extract face from: {image_path}")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        print(f"Successfully loaded {len(self.reference_embeddings)} reference faces.")
    
    def extract_face_embedding_from_path(self, image_path):
        """Extract embedding from an image file path"""
        try:
            # Align face from image path
            aligned_face = simple_face_align(image_path)
            
            # Preprocess the aligned face
            transformed_input = self.transform(aligned_face)
            
            # Add batch dimension
            transformed_input = transformed_input.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(transformed_input)
            
            return embedding.squeeze().numpy()
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def extract_face_embedding_from_crop(self, face_crop):
        """Extract embedding from a face crop (numpy array)"""
        try:
            # Align face
            aligned_face = simple_face_align(face_crop)
            
            # Preprocess the aligned face
            transformed_input = self.transform(aligned_face)
            
            # Add batch dimension
            transformed_input = transformed_input.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(transformed_input)
            
            return embedding.squeeze().numpy()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def identify_face(self, face_embedding):
        """Compare face embedding with reference embeddings"""
        if not self.reference_embeddings:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for person_name, ref_embedding in self.reference_embeddings.items():
            similarity = cosine_similarity(face_embedding, ref_embedding)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = person_name
        
        return best_match, best_similarity
    
    def detect_and_recognize_faces(self, frame):
        """Detect faces in frame and identify them"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        face_identities = []
        face_locations = []
        
        for (x, y, w, h) in faces:
            # Add padding around face
            padding = int(0.2 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            # Crop face
            face_crop = frame[y_start:y_end, x_start:x_end]
            
            # Extract embedding
            embedding = self.extract_face_embedding_from_crop(face_crop)
            
            if embedding is not None:
                # Identify the face
                identity, confidence = self.identify_face(embedding)
                face_identities.append((identity, confidence))
                face_locations.append((x, y, w, h))
            else:
                face_identities.append(("Error", 0.0))
                face_locations.append((x, y, w, h))
        
        return face_identities, face_locations
    
    def draw_face_boxes(self, frame, face_locations, face_identities):
        """Draw bounding boxes around detected faces with identity labels"""
        for i, ((x, y, w, h), (identity, confidence)) in enumerate(zip(face_locations, face_identities)):
            # Choose color based on recognition status
            if identity == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            elif identity == "Error":
                color = (0, 255, 255)  # Yellow for error
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            if identity != "Error":
                label = f'{identity} ({confidence:.2f})'
            else:
                label = "Error"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop for real-time face recognition"""
        fps_counter = 0
        start_time = time.time()
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces and identify them
            face_identities, face_locations = self.detect_and_recognize_faces(frame)
            
            # Draw face boxes and labels
            frame_with_boxes = self.draw_face_boxes(frame, face_locations, face_identities)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
                print(f"FPS: {fps:.2f} | Faces detected: {len(face_locations)}")
            
            # Add instructions to frame
            instructions = [
                "Press 'q' to quit, 'r' to reload images",
                f"Reference faces: {len(self.reference_embeddings)}",
                f"Threshold: {self.similarity_threshold}"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame_with_boxes, instruction, 
                           (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-time Face Recognition', frame_with_boxes)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading reference images...")
                self.load_reference_images()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Face recognition system stopped.")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the real-time face recognition system
        # You can adjust the similarity threshold (0.0 to 1.0)
        # Higher threshold = more strict matching
        face_recognizer = RealTimeFaceRecognition(
            model_name="edgeface_s_gamma_05",
            images_folder="images",  # Folder containing reference face images
            similarity_threshold=0.6
        )
        
        # Run the system
        face_recognizer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is working and the model files are in the correct location.")