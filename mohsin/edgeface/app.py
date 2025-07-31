from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import time
import sys
import json
import numpy as np
import speech_recognition as sr
import threading
from queue import Queue
from test import RealTimeFaceRecognition, simple_face_align

# Add project root to Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Import functions from new_test
from aryan.new_test import get_weather, build_prompt, call_groq, load_user_data

# Import the backbones module from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

app = Flask(__name__)

# Initialize face recognizer
face_recognizer = RealTimeFaceRecognition(
    model_name="edgeface_s_gamma_05",
    images_folder=os.path.join(current_dir, 'images'),
    similarity_threshold=0.6
)

# Profile folder path
PROFILES_DIR = os.path.join(PROJECT_ROOT, 'mohsin', 'profiles')
USER_DATA_DIR = os.path.join(PROJECT_ROOT, 'user_data')

# Store the latest clothing response and weather info
app.config['clothing_response'] = ""
app.config['weather_info'] = {}
app.config['outfit_recommendation'] = {}
app.config['mirror_active'] = False
app.config['camera_active'] = False

# Initialize speech recognition
recognizer = sr.Recognizer()
voice_command_queue = Queue()

def initialize_camera():
    """Initialize the camera and face recognizer"""
    global face_recognizer
    face_recognizer.cap = cv2.VideoCapture(0)
    if face_recognizer.cap.isOpened():
        face_recognizer.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        face_recognizer.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
        app.config['camera_active'] = True
        print("Camera initialized successfully")
    else:
        print("Failed to initialize camera")
        app.config['camera_active'] = False

def shutdown_camera():
    """Shutdown the camera"""
    global face_recognizer
    if hasattr(face_recognizer, 'cap') and face_recognizer.cap is not None:
        face_recognizer.cap.release()
    app.config['camera_active'] = False
    print("Camera shut down")

def listen_for_wake_word():
    """Continuously listen for the wake word 'hello mirror'"""
    while True:
        with sr.Microphone() as source:
            try:
                print("Listening for 'hello'...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    if "hello" in text and not app.config['mirror_active']:
                        print("Wake word detected! Activating mirror...")
                        app.config['mirror_active'] = True
                        
                        # Initialize camera for recognition
                        print("Opening camera for recognition...")
                        initialize_camera()
                        
                        # Wait for 5 seconds while trying to recognize the person
                        recognition_start = time.time()
                        while time.time() - recognition_start < 5:
                            ret, frame = face_recognizer.cap.read()
                            if ret:
                                face_identities, _ = face_recognizer.detect_and_recognize_faces(frame)
                                if face_identities and face_identities[0][0] not in ("Unknown", "Error"):
                                    identity = face_identities[0][0]
                                    app.config['last_identity'] = identity
                                    print(f"Person recognized: {identity}")
                                    break
                            time.sleep(0.1)
                        
                        # Shutdown camera after recognition
                        shutdown_camera()
                        
                        # Get weather info after recognition
                        try:
                            temp, season, condition, additional_prep = get_weather()
                            app.config['weather_info'] = {
                                'temp': temp,
                                'season': season,
                                'condition': condition,
                                'additional_prep': additional_prep
                            }
                        except Exception as e:
                            print(f"Error getting weather: {e}")
                    
                    elif "i want to go to" in text and app.config['mirror_active']:
                        print("Event request detected!")
                        event = "office" if "office" in text else "casual"  # Default to casual if not office
                        # Get outfit recommendation directly
                        try:
                            identity = app.config.get('last_identity', 'Unknown')
                            if identity not in ('Unknown', 'Error'):
                                # Get outfit recommendation
                                personal_info, wardrobe = load_user_data(identity.lower())
                                temp, season, condition, additional_prep = get_weather()
                                prompt = build_prompt(event, temp, season, condition, personal_info, wardrobe)
                                response = call_groq(prompt)
                                
                                # Parse response and update app config
                                lines = response.strip().split('\n')
                                ids_line = lines[0]
                                explanation = ' '.join(lines[1:]).strip()
                                
                                # Get outfit details
                                outfit_items = []
                                for item_id in [id_.strip() for id_ in ids_line.split(',') if id_.strip().isdigit()]:
                                    item = next((w for w in wardrobe if w["item_id"] == item_id), None)
                                    if item:
                                        outfit_items.append(item)
                                
                                app.config['outfit_recommendation'] = {
                                    'outfit_items': outfit_items,
                                    'explanation': explanation,
                                    'additional_prep': additional_prep
                                }
                                print(f"Got outfit recommendation for {event}")
                            else:
                                print("No person recognized, cannot get outfit recommendation")
                        except Exception as e:
                            print(f"Error getting outfit recommendation: {e}")
                    
                    elif "goodbye mirror" in text and app.config['mirror_active']:
                        print("Deactivating mirror...")
                        app.config['mirror_active'] = False
                        app.config['weather_info'] = {}
                        app.config['outfit_recommendation'] = {}
                        shutdown_camera()  # Stop the camera
                        
                except sr.UnknownValueError:
                    pass  # Speech was unclear
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    
            except sr.WaitTimeoutError:
                pass  # No speech detected
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(1)

def find_profile_folder(person_name):
    profile_path = os.path.join(PROFILES_DIR, person_name)
    if os.path.exists(profile_path) and os.path.isdir(profile_path):
        return profile_path
    return None

def gen_frames():
    """Generator function to stream video frames with face recognition and overlaid text"""
    blank_frame = None
    
    while True:
        try:
            # Create a blank frame for displaying information
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            height, width = frame.shape[:2]

            if not app.config['mirror_active']:
                # Display activation instruction
                text = "Listening for 'Hello'..."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = height // 2
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                          (255, 255, 255), thickness, cv2.LINE_AA)
            else:
                # Get identity
                identity = app.config.get('last_identity', 'Unknown')
                if identity not in ('Unknown', 'Error'):
                    # Display greeting in the center top
                    greeting_text = f"Hello, {identity}"
                    font_scale = 1.5
                    thickness = 2
                    text_size = cv2.getTextSize(greeting_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (width - text_size[0]) // 2
                    cv2.putText(frame, greeting_text, 
                              (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                # Display weather info if available (centered below greeting)
                if app.config['weather_info']:
                    weather = app.config['weather_info']
                    weather_text = f"{weather.get('temp', 'N/A')}°C - {weather.get('condition', 'N/A')}"
                    font_scale = 1.2
                    thickness = 2
                    text_size = cv2.getTextSize(weather_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (width - text_size[0]) // 2
                    cv2.putText(frame, weather_text, 
                              (text_x, 180), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            if app.config['mirror_active']:
                # Display outfit recommendation if available
                if app.config['outfit_recommendation']:
                    outfit = app.config['outfit_recommendation']
                    if outfit.get('outfit_items'):
                        # Display the recommendation centered on screen
                        y_position = height // 2 + 50  # Start below the weather info
                        
                        # Display each item on a new line
                        for item in outfit['outfit_items']:
                            item_text = f"{item['type']}: {item['color']} {item.get('texture', '')} {item.get('pattern', '')}".strip()
                            text_size = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            text_x = (width - text_size[0]) // 2
                            cv2.putText(frame, item_text, 
                                      (text_x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 255, 0), 2, cv2.LINE_AA)
                            y_position += 40
                # Display voice command instructions at bottom
                if app.config['mirror_active']:
                    instructions1 = "Say 'I want to go to office' for outfit recommendation"
                    instructions2 = "Say 'Goodbye Mirror' to deactivate"
                    
                    text_size1 = cv2.getTextSize(instructions1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_size2 = cv2.getTextSize(instructions2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    text_x1 = (width - text_size1[0]) // 2
                    text_x2 = (width - text_size2[0]) // 2
                    
                    cv2.putText(frame, instructions1, 
                              (text_x1, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(frame, instructions2, 
                              (text_x2, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (200, 200, 200), 1, cv2.LINE_AA)            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame, retrying...")
                time.sleep(0.1)
                continue
            
            frame = buffer.tobytes()
            
            # Yield frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            print(f"Error in video feed: {e}, retrying...")
            time.sleep(0.1)
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_weather_info', methods=['GET'])
def weather_info():
    identity = app.config.get('last_identity', 'Unknown')
    
    if identity in ('Unknown', 'Error'):
        return jsonify({
            'error': 'No person recognized'
        }), 404

    try:
        # Get weather information
        temp, season, condition, additional_prep = get_weather()
        
        # Store weather info in app config
        weather_info = {
            'temp': temp,
            'season': season,
            'condition': condition,
            'additional_prep': additional_prep
        }
        app.config['weather_info'] = weather_info
        
        return jsonify(weather_info)
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to get weather information: {str(e)}'
        }), 500

@app.route('/get_outfit_recommendation', methods=['POST'])
def get_outfit_recommendation():
    identity = app.config.get('last_identity', 'Unknown')
    event = request.form.get('event')
    
    if not event:
        return jsonify({
            'error': 'Event type is required'
        }), 400
    
    if identity in ('Unknown', 'Error'):
        return jsonify({
            'error': 'No person recognized'
        }), 404
        
    try:
        # Load user data
        personal_info, wardrobe = load_user_data(identity.lower())
        
        # Get weather info
        temp, season, condition, additional_prep = get_weather()
        
        # Build and send prompt to get outfit recommendation
        prompt = build_prompt(event, temp, season, condition, personal_info, wardrobe)
        response = call_groq(prompt)
        
        # Parse response
        lines = response.strip().split('\n')
        ids_line = lines[0]
        explanation = ' '.join(lines[1:]).strip()
        
        # Get outfit details
        outfit_items = []
        for item_id in [id_.strip() for id_ in ids_line.split(',') if id_.strip().isdigit()]:
            item = next((w for w in wardrobe if w["item_id"] == item_id), None)
            if item:
                outfit_items.append(item)
        
        recommendation = {
            'outfit_items': outfit_items,
            'explanation': explanation,
            'additional_prep': additional_prep
        }
        
        # Store recommendation in app config
        app.config['outfit_recommendation'] = recommendation
        
        return jsonify(recommendation)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get outfit recommendation: {str(e)}'
        }), 500

@app.route('/submit_clothing', methods=['POST'])
def submit_clothing():
    clothing = request.form.get('clothing')
    identity = app.config.get('last_identity', 'Unknown')
    
    if identity and identity != "Unknown" and identity != "Error":
        profile_folder = find_profile_folder(identity)
        if profile_folder:
            response = f"{identity} wants to wear: {clothing}"
        else:
            response = f"No profile folder found for {identity}"
    else:
        response = "No person recognized"
    
    # Store the response to display on the video
    app.config['clothing_response'] = response
    return jsonify({'response': response})

@app.teardown_appcontext
def cleanup(exception=None):
    """Release webcam when app shuts down"""
    shutdown_camera()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the voice recognition in a separate thread
    voice_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    voice_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)