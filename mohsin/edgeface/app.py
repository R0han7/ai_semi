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
import pyrealsense2 as rs
import requests
from datetime import datetime
from test import RealTimeFaceRecognition, simple_face_align

# Add project root to Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Import functions from new_test
from aryan.new_test import build_prompt, call_groq, load_user_data

# Import the backbones module from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

app = Flask(__name__)

# Weather API Configuration
LAT = 33.6844  # Kitakyushu latitude
LON = 130.4017  # Kitakyushu longitude
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"

# RealSense pipeline and configuration
pipeline = None
config = None
align = None

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

def get_weather_json():
    """Enhanced weather API function with detailed hourly data"""
    try:
        # Weather API request
        params = {
            "latitude": LAT,
            "longitude": LON,
            "current_weather": True,
            "hourly": "uv_index,precipitation,relative_humidity_2m,windspeed_10m,temperature_2m",
            "timezone": "auto"
        }
        weather_res = requests.get(WEATHER_API_URL, params=params, timeout=5)
        weather_res.raise_for_status()
        weather_data = weather_res.json()

        # AQI API request
        aqi_res = requests.get(AQI_API, params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": "european_aqi",
            "timezone": "auto"
        }, timeout=5)
        aqi_res.raise_for_status()
        aqi_data = aqi_res.json()

        # Convert API times to datetime and find latest valid timestamp up to now
        now = datetime.now()
        weather_times = weather_data["hourly"]["time"]
        valid_indices = [
            i for i, t in enumerate(weather_times)
            if datetime.fromisoformat(t) <= now
        ]

        if not valid_indices:
            raise Exception("No valid past timestamps available.")

        latest_index = valid_indices[-1]
        latest_time = weather_times[latest_index]

        # AQI might not align perfectly — find matching time if available
        aqi_times = aqi_data["hourly"]["time"]
        aqi_index = next((i for i, t in enumerate(aqi_times) if t == latest_time), None)
        aqi = aqi_data["hourly"]["european_aqi"][aqi_index] if aqi_index is not None else "N/A"

        # Extract values
        temperature = weather_data["hourly"]["temperature_2m"][latest_index]
        windspeed = weather_data["hourly"]["windspeed_10m"][latest_index]
        humidity = weather_data["hourly"]["relative_humidity_2m"][latest_index]
        uv_index = weather_data["hourly"]["uv_index"][latest_index]
        precipitation = weather_data["hourly"]["precipitation"][latest_index]

        # Get current weather info for season and condition determination
        current_weather = weather_data.get('current_weather', {})
        weather_code = current_weather.get('weathercode', 0)
        is_day = current_weather.get('is_day', 1)

        # Convert weather code to readable condition
        condition = get_weather_condition_from_code(weather_code)
        
        # Determine season from temperature
        season = determine_season_from_temp(temperature)
        
        # Get additional preparation suggestions
        additional_prep = get_additional_weather_prep(weather_code, temperature, bool(is_day), precipitation, uv_index, windspeed)

        # Display weather info
        print("\n🌤️ Current Weather Conditions (Most Recent Hour):")
        print(f"Time: {latest_time}")
        print(f"Temperature: {temperature} °C")
        print(f"Humidity: {humidity} %")
        print(f"UV Index: {uv_index}")
        print(f"Windspeed: {windspeed} km/h")
        print(f"Precipitation: {precipitation} mm")
        print(f"Air Quality Index (AQI): {aqi}")

        return {
            'time': latest_time,
            'temp': temperature,
            'humidity': humidity,
            'uv_index': uv_index,
            'windspeed': windspeed,
            'precipitation': precipitation,
            'aqi': aqi,
            'condition': condition,
            'season': season,
            'additional_prep': additional_prep,
            'is_day': bool(is_day),
            'weather_code': weather_code
        }

    except Exception as e:
        print(f"⚠️ Weather API fetch failed: {e}")
        # Return default values in case of error
        return {
            'time': 'N/A',
            'temp': 'N/A',
            'humidity': 'N/A',
            'uv_index': 'N/A',
            'windspeed': 'N/A',
            'precipitation': 'N/A',
            'aqi': 'N/A',
            'condition': 'Unknown',
            'season': 'unknown',
            'additional_prep': 'Weather data unavailable',
            'is_day': True,
            'weather_code': 0
        }

def get_weather_condition_from_code(weather_code):
    """Convert WMO weather code to readable condition"""
    weather_conditions = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
    }
    return weather_conditions.get(weather_code, "Unknown")

def determine_season_from_temp(temperature):
    """Determine season based on temperature (rough estimation)"""
    if temperature < 5:
        return "winter"
    elif temperature < 15:
        return "spring/autumn"
    elif temperature < 25:
        return "mild"
    else:
        return "summer"

def get_additional_weather_prep(weather_code, temperature, is_day, precipitation=0, uv_index=0, windspeed=0):
    """Enhanced preparation suggestions based on detailed weather conditions"""
    prep_suggestions = []
    
    # Temperature-based suggestions
    if temperature < 5:
        prep_suggestions.append("Very cold - wear heavy winter clothing")
    elif temperature < 10:
        prep_suggestions.append("Bring a warm coat")
    elif temperature > 30:
        prep_suggestions.append("Very hot - stay hydrated and seek shade")
    elif temperature > 25:
        prep_suggestions.append("Warm weather - light clothing recommended")
    
    # Precipitation-based suggestions
    if precipitation > 0:
        if precipitation > 5:
            prep_suggestions.append("Heavy rain expected - bring umbrella and waterproof clothing")
        else:
            prep_suggestions.append("Light rain possible - bring an umbrella")
    
    # Weather code-based suggestions
    if weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:  # Rain/drizzle
        prep_suggestions.append("Rainy conditions - non-slip shoes recommended")
    elif weather_code in [71, 73, 75, 85, 86]:  # Snow
        prep_suggestions.append("Snow conditions - wear warm, non-slip shoes")
    elif weather_code in [45, 48]:  # Fog
        prep_suggestions.append("Foggy conditions - drive carefully, low visibility")
    elif weather_code in [95, 96, 99]:  # Thunderstorm
        prep_suggestions.append("Thunderstorm warning - avoid outdoor activities")
    
    # UV Index suggestions
    if uv_index > 7:
        prep_suggestions.append("High UV - wear sunscreen and sunglasses")
    elif uv_index > 3:
        prep_suggestions.append("Moderate UV - consider sun protection")
    
    # Wind-based suggestions
    if windspeed > 25:
        prep_suggestions.append("Very windy - secure loose items")
    elif windspeed > 15:
        prep_suggestions.append("Windy conditions - dress accordingly")
    
    # Day/night suggestions
    if not is_day:
        prep_suggestions.append("Nighttime - bring flashlight or use phone light")
    
    return "; ".join(prep_suggestions) if prep_suggestions else "No special preparation needed"

def initialize_realsense_camera():
    """Initialize the Intel RealSense camera"""
    global pipeline, config, align
    
    try:
        # Create a pipeline
        pipeline = rs.pipeline()
        
        # Create a config and configure the pipeline to stream
        config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            return False
        
        # Configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        
        # Create an align object for aligning depth to color
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        app.config['camera_active'] = True
        print("RealSense camera initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize RealSense camera: {e}")
        app.config['camera_active'] = False
        return False

def get_realsense_frame():
    """Get a frame from the RealSense camera"""
    global pipeline, align
    
    if pipeline is None:
        return None, None
    
    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned color and depth frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return None, None
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
        
    except Exception as e:
        print(f"Error getting RealSense frame: {e}")
        return None, None

def shutdown_realsense_camera():
    """Shutdown the RealSense camera"""
    global pipeline
    
    try:
        if pipeline is not None:
            pipeline.stop()
            pipeline = None
        app.config['camera_active'] = False
        print("RealSense camera shut down")
    except Exception as e:
        print(f"Error shutting down RealSense camera: {e}")

# Modified RealTimeFaceRecognition class to work with RealSense
class RealSenseFaceRecognition(RealTimeFaceRecognition):
    def __init__(self, model_name, images_folder, similarity_threshold=0.6):
        super().__init__(model_name, images_folder, similarity_threshold)
        # Remove the cap attribute since we're using RealSense
        if hasattr(self, 'cap'):
            delattr(self, 'cap')
    
    def get_frame(self):
        """Get frame from RealSense camera instead of OpenCV"""
        return get_realsense_frame()
    
    def detect_and_recognize_faces_realsense(self, color_frame, depth_frame=None):
        """Modified face detection that works with RealSense frames"""
        if color_frame is None:
            return [], []
        
        # Use the existing face detection logic but with RealSense frame
        return self.detect_and_recognize_faces(color_frame)

# Update face recognizer to use RealSense version
face_recognizer = RealSenseFaceRecognition(
    model_name="edgeface_s_gamma_05",
    images_folder=os.path.join(current_dir, 'images'),
    similarity_threshold=0.6
)

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
                        
                        # Initialize RealSense camera for recognition
                        print("Opening RealSense camera for recognition...")
                        if initialize_realsense_camera():
                            
                            # Wait for 5 seconds while trying to recognize the person
                            recognition_start = time.time()
                            while time.time() - recognition_start < 5:
                                color_frame, depth_frame = get_realsense_frame()
                                if color_frame is not None:
                                    face_identities, _ = face_recognizer.detect_and_recognize_faces_realsense(
                                        color_frame, depth_frame
                                    )
                                    if face_identities and face_identities[0][0] not in ("Unknown", "Error"):
                                        identity = face_identities[0][0]
                                        app.config['last_identity'] = identity
                                        print(f"Person recognized: {identity}")
                                        break
                                time.sleep(0.1)
                            
                            # Shutdown camera after recognition
                            shutdown_realsense_camera()
                        else:
                            print("Failed to initialize RealSense camera")
                        
                        # Get weather info after recognition using enhanced API
                        try:
                            weather_info = get_weather_json()
                            app.config['weather_info'] = weather_info
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
                                weather_info = get_weather_json()
                                prompt = build_prompt(event, weather_info['temp'], weather_info['season'], weather_info['condition'], personal_info, wardrobe)
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
                                    'additional_prep': weather_info['additional_prep'],
                                    'weather_info': weather_info
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
                        shutdown_realsense_camera()  # Stop the RealSense camera
                        
                except sr.UnknownValueError:
                    pass  # Speech was unclear
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    
            except sr.WaitTimeoutError:
                pass  # No speech detected
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(1)

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
                
                # Display enhanced weather info if available
                if app.config['weather_info']:
                    weather = app.config['weather_info']
                    print(f"Displaying weather info: {weather}")
                    
                    y_pos = 180
                    font_scale = 0.8
                    
                    # Temperature and condition
                    if weather.get('temp') != 'N/A':
                        temp_text = f"Temperature: {weather['temp']}C - {weather.get('condition', 'N/A')}"
                        temp_text=temp_text.replace("?", " ")
                        text_size = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, temp_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (255, 255, 255), 2, cv2.LINE_AA)
                        y_pos += 35
                    
                    # Humidity
                    if weather.get('humidity') != 'N/A':
                        humidity_text = f"Humidity: {weather['humidity']}%"
                        text_size = cv2.getTextSize(humidity_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, humidity_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                        y_pos += 30
                    
                    # UV Index
                    if weather.get('uv_index') != 'N/A':
                        uv_text = f"UV Index: {weather['uv_index']}"
                        text_size = cv2.getTextSize(uv_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, uv_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                        y_pos += 30
                    
                    # Wind speed
                    if weather.get('windspeed') != 'N/A':
                        wind_text = f"Wind: {weather['windspeed']} km/h"
                        text_size = cv2.getTextSize(wind_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, wind_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                        y_pos += 30
                    
                    # Precipitation
                    if weather.get('precipitation') != 'N/A':
                        precip_text = f"Precipitation: {weather['precipitation']} mm"
                        text_size = cv2.getTextSize(precip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, precip_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                        y_pos += 30
                    
                    # Air Quality Index
                    if weather.get('aqi') != 'N/A':
                        aqi_text = f"Air Quality Index: {weather['aqi']}"
                        text_size = cv2.getTextSize(aqi_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                        text_x = (width - text_size[0]) // 2
                        cv2.putText(frame, aqi_text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            if app.config['mirror_active']:
                # Display outfit recommendation if available
                if app.config['outfit_recommendation']:
                    outfit = app.config['outfit_recommendation']
                    if outfit.get('outfit_items'):
                        # Display the recommendation centered on screen
                        y_position = height // 2 + 100  # Start below the weather info
                        
                        # Display each item on a new line
                        for item in outfit['outfit_items']:
                            item_text = f"{item['type']}: {item['color']} {item.get('texture', '')} {item.get('pattern', '')}".strip()
                            text_size = cv2.getTextSize(item_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            text_x = (width - text_size[0]) // 2
                            cv2.putText(frame, item_text, 
                                      (text_x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 255, 0), 2, cv2.LINE_AA)
                            y_position += 40
                            
                        # Display additional weather preparation if available
                        if outfit.get('additional_prep') and outfit['additional_prep'] != "No special preparation needed":
                            y_position += 20  # Add some space
                            prep_text = f"Weather tip: {outfit['additional_prep']}"
                            # Split long text into multiple lines if needed
                            max_width = width - 100
                            words = prep_text.split()
                            line = ""
                            for word in words:
                                test_line = line + word + " "
                                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                                if text_size[0] <= max_width:
                                    line = test_line
                                else:
                                    if line:
                                        text_x = (width - cv2.getTextSize(line.strip(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]) // 2
                                        cv2.putText(frame, line.strip(), 
                                                  (text_x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.6, (255, 255, 0), 1, cv2.LINE_AA)
                                        y_position += 30
                                    line = word + " "
                            if line:
                                text_x = (width - cv2.getTextSize(line.strip(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]) // 2
                                cv2.putText(frame, line.strip(), 
                                          (text_x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (255, 255, 0), 1, cv2.LINE_AA)
                
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
                              0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
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

def find_profile_folder(person_name):
    profile_path = os.path.join(PROFILES_DIR, person_name)
    if os.path.exists(profile_path) and os.path.isdir(profile_path):
        return profile_path
    return None

# Optional: Add route to get depth information
@app.route('/depth_info')
def get_depth_info():
    """Get depth information from the current frame"""
    if not app.config['camera_active']:
        return jsonify({'error': 'Camera not active'})
    
    color_frame, depth_frame = get_realsense_frame()
    if depth_frame is not None:
        # Get depth statistics
        depth_stats = {
            'min_depth': float(np.min(depth_frame)),
            'max_depth': float(np.max(depth_frame)),
            'mean_depth': float(np.mean(depth_frame)),
            'center_depth': float(depth_frame[depth_frame.shape[0]//2, depth_frame.shape[1]//2])
        }
        return jsonify(depth_stats)
    else:
        return jsonify({'error': 'No depth data available'})

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
        # Get enhanced weather information
        weather_info = get_weather_json()
        
        # Store weather info in app config
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
        
        # Get enhanced weather info
        weather_info = get_weather_json()
        
        # Build and send prompt to get outfit recommendation using enhanced weather data
        prompt = build_prompt(
            event, 
            weather_info['temp'], 
            weather_info['season'], 
            weather_info['condition'], 
            personal_info, 
            wardrobe
        )
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
            'additional_prep': weather_info['additional_prep'],
            'weather_info': weather_info  # Include full enhanced weather info
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
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the voice recognition in a separate thread
    voice_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    voice_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)