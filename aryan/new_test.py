import os
import re
import json
import random
import glob
import requests
import pytz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as PILImage
import numpy as np
from datetime import datetime

# -------------------- CONFIG --------------------
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = "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"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 33.8888387802569, 130.71054525598575
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")
USER_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "user_data")
DEBUG = True  # Toggle debug prints

# -------------------- USER DATA --------------------
def load_user_data(username):
    user_dir = os.path.join(USER_DATA_DIR, username)
    if not os.path.exists(user_dir):
        raise ValueError(f"User directory not found for {username}")
    
    # Load personal info
    personal_info_path = os.path.join(user_dir, "personal_info.json")
    if not os.path.exists(personal_info_path):
        raise ValueError(f"Personal info file not found for {username}")
    with open(personal_info_path, 'r') as f:
        personal_info = json.load(f)
    
    # Load wardrobe
    wardrobe_path = os.path.join(user_dir, "wardrobe.json")
    if not os.path.exists(wardrobe_path):
        raise ValueError(f"Wardrobe file not found for {username}")
    with open(wardrobe_path, 'r') as f:
        wardrobe = json.load(f)
    
    return personal_info, wardrobe

# -------------------- WEATHER --------------------
def get_weather():
    try:
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
        weather = weather_data["current_weather"]
        temp = weather["temperature"]
        code = weather["weathercode"]
        hour = datetime.now().hour
        uv_index = weather_data["hourly"]["uv_index"][hour]
        precipitation_next_2h = max(weather_data["hourly"]["precipitation"][hour:hour+3])
        humidity = weather_data["hourly"]["relative_humidity_2m"][hour]
        wind_speed = weather_data["hourly"]["windspeed_10m"][hour]
        morning_temp = weather_data["hourly"]["temperature_2m"][6] if "temperature_2m" in weather_data["hourly"] else temp

        AQI_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_res = requests.get(AQI_API, params={
            "latitude": LAT, "longitude": LON, "hourly": "european_aqi", "timezone": "auto"
        }, timeout=5)
        aqi_res.raise_for_status()
        aqi_data = aqi_res.json()
        aqi = aqi_data["hourly"]["european_aqi"][hour]

        if DEBUG:
            print("\n📊 Weather & Health Data:")
            print(f"UV Index: {uv_index}")
            print(f"Precipitation (next 2h): {precipitation_next_2h} mm")
            print(f"Humidity: {humidity}%")
            print(f"Wind Speed: {wind_speed} km/h")
            print(f"AQI: {aqi}")
            print(f"Morning Temp (6 AM): {morning_temp}°C | Current Temp: {temp}°C\n")

        if 0 <= code < 10:
            condition = "clear"
        elif 10 <= code < 20:
            condition = "partly cloudy"
        elif 20 <= code < 30:
            condition = "cloudy"
        elif 30 <= code < 40:
            condition = "foggy"
        elif 50 <= code < 60:
            condition = "rainy"
        elif 60 <= code < 70:
            condition = "stormy"
        elif 70 <= code < 80:
            condition = "snowy"
        else:
            condition = "unknown"

        if condition in ["clear", "partly cloudy", "cloudy", "foggy"]:
            if temp < 10:
                condition += " & cold"
            elif temp > 30:
                condition += " & hot"
            else:
                condition += " & mild"

        month = datetime.now(pytz.utc).month
        season = (
            "spring" if 3 <= month <= 5 else
            "summer" if 6 <= month <= 8 else
            "autumn" if 9 <= month <= 11 else
            "winter"
        )

        recs = []
        if uv_index >= 6:
            recs.append("high UV: apply sunscreen, wear sunglasses and a hat")
        if precipitation_next_2h >= 0.5:
            recs.append("chance of rain: carry an umbrella")
        if humidity >= 70 or temp >= 30:
            recs.append("hot/humid: drink plenty of water")
        if wind_speed >= 25:
            recs.append("windy: wear a windbreaker, consider tying your hair")
        if aqi >= 100:
            recs.append("poor air quality: carry a mask")
        if abs(temp - morning_temp) >= 10:
            recs.append("layer clothing for temperature changes")

        additional_preparation = ", ".join(recs) if recs else "no additional preparation needed"
        return temp, season, condition, additional_preparation

    except Exception as e:
        print(f"⚠️ Weather fetch failed: {e}")
        return "unknown", "unknown", "unknown", "unknown"

# -------------------- PROMPT BUILD --------------------
def build_prompt(event, temp, season, condition, personal_info, wardrobe):
    items = "\n".join([
        f"- {w['item_id']}: {w['color']} {w['texture']} {w['pattern']} {w['type']} (season: {w['season']}, formality: {w['formality']})"
        for w in wardrobe
    ])
    return f"""
You are a professional AI personal stylist for a high-end fashion assistant service.

You will receive:
- The user's personal details
- A complete wardrobe inventory (color, texture, pattern, type, season, formality)
- The upcoming event details
- Current weather, temperature, and season

Your task:
✅ Select the best one-day full outfit (top, bottom, shoes) for the user, appropriate for the weather, event, and season.
✅ Follow color theory principles (complementary, analogous, trending colors) to ensure the outfit is visually appealing and stylish.
✅ Ensure seasonal and weather-appropriate choices (e.g., no sandals during rain, no heavy jackets in summer).
✅ If the event is formal (e.g., wedding, gala, conference), prioritize formal or semi-formal items for tops and shoes.
✅ Ensure the colors harmonize with the user's skin tone, body type, and face shape to enhance their appearance.

FIRST, output the three chosen item IDs separated by commas on the first line (e.g., 003, 004, 006).
THEN, in one or two sentences, explain why this outfit was chosen, referencing weather, event, and color harmony.
Do not add any additional text outside these instructions.

User details:
- Height: {personal_info['height']}
- Weight: {personal_info['weight']}
- Skin tone: {personal_info['skin_tone']}
- Body type: {personal_info['body_type']}
- Face shape: {personal_info['face_shape']}
- Preferred color scheme: {personal_info['preferred_color_scheme']}

Wardrobe items:
{items}

Event: {event}
Current temperature: {temp}°C
Current season: {season}
Weather condition: {condition}
"""

# -------------------- GROQ CALL --------------------
def call_groq(prompt):
    res = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        data=json.dumps({
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }),
        timeout=10
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

# -------------------- DISPLAY OUTFIT --------------------
def display_outfit(ids, wardrobe):
    id_list = [id_.strip() for id_ in ids.split(",") if id_.strip().isdigit()]
    print("\n🎉 Recommended Outfit:")
    for id_ in id_list:
        item = next((w for w in wardrobe if w["item_id"] == id_), None)
        if item:
            formality = item["formality"]
            if isinstance(formality, list):
                formality = "/".join(formality)
            print(f"✅ {item['type'].capitalize()}: {item['color']} {item['texture']} {item['pattern']} (season: {item['season']}, formality: {formality})")
        else:
            print(f"⚠️ Unknown item ID: {id_}")
    return id_list

# -------------------- DISPLAY IMAGES --------------------
def display_images(outfit_ids, wardrobe):
    imgs, titles = [], []
    for id_ in outfit_ids:
        item = next((w for w in wardrobe if w["item_id"] == id_), None)
        if item:
            matches = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.avif"]:
                matches += glob.glob(os.path.join(IMAGE_DIR, f"{item['base']}{ext}"))
            if not matches:
                imgs.append(None)
                titles.append(f"{item['type'].title()}\n{item['color']}\n{item['formality']}")
                continue
            path = matches[0]
            ext = os.path.splitext(path)[1].lower()
            img = np.array(PILImage.open(path)) if ext == ".avif" else mpimg.imread(path)

            formality = item["formality"]
            if isinstance(formality, list):
                formality = "/".join(formality)
            titles.append(f"{item['type'].title()}\n{item['color']}\n{formality}")
            imgs.append(img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, imgs, titles):
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()

# -------------------- MAIN --------------------
def main():
    # Get username
    username = input("Enter username: ").strip()
    
    try:
        # Load user data
        personal_info, wardrobe = load_user_data(username)
    except ValueError as e:
        print(f"Error: {e}")
        return

    event = input("Enter event (e.g., office party, wedding, casual outing): ").strip()
    temp, season, condition, additional_prep = get_weather()
    print(f"\n🌤 Temp: {temp}°C | Season: {season} | Condition: {condition}\n")

    prompt = build_prompt(event, temp, season, condition, personal_info, wardrobe)
    
    response = call_groq(prompt)
   
    lines = response.strip().split("\n")
    ids_line = lines[0]
    explanation = " ".join(lines[1:]).strip()

    id_list = display_outfit(ids_line, wardrobe)
    display_images(id_list, wardrobe)

    if explanation:
        print(f"\n💡 Why this outfit?\n{explanation}\n")
    if additional_prep and additional_prep != "no additional preparation needed":
        print(f"\n🌿 Additional day preparation:\n{additional_prep}\n")

if __name__ == "__main__":
    main()
