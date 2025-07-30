from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import time
import sys
import json
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

def find_profile_folder(person_name):
    profile_path = os.path.join(PROFILES_DIR, person_name)
    if os.path.exists(profile_path) and os.path.isdir(profile_path):
        return profile_path
    return None

def gen_frames():
    """Generator function to stream video frames with face recognition and overlaid text"""
    while True:
        try:
            # Check if webcam is open, reinitialize if not
            if not face_recognizer.cap.isOpened():
                print("Webcam disconnected, attempting to reinitialize...")
                face_recognizer.cap = cv2.VideoCapture(0)
                if not face_recognizer.cap.isOpened():
                    print("Failed to open webcam, retrying in 1 second...")
                    time.sleep(1)
                    continue
                face_recognizer.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                face_recognizer.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

            # Capture frame
            ret, frame = face_recognizer.cap.read()
            if not ret:
                print("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect and recognize faces
            face_identities, face_locations = face_recognizer.detect_and_recognize_faces(frame)
            
            # Get the latest identity
            identity = "Unknown"
            confidence = 0.0
            if face_identities and face_identities[0][0] not in ("Unknown", "Error"):
                identity = face_identities[0][0]
                confidence = face_identities[0][1]
            app.config['last_identity'] = identity

            # Overlay text on the frame
            # Display recognized identity
            cv2.putText(frame, f"Recognized: {identity} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display weather info if available
            if app.config['weather_info']:
                weather = app.config['weather_info']
                cv2.putText(frame, f"Weather: {weather['condition']} {weather['temp']}°C", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display outfit recommendation if available
            if app.config['outfit_recommendation']:
                outfit = app.config['outfit_recommendation']
                cv2.putText(frame, f"Recommended: {outfit.get('explanation', '')}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode frame as JPEG
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
    face_recognizer.cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)