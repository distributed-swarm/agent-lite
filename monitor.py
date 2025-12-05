import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw
import requests
import time
import threading

# Configuration
CONTROLLER_URL = "http://localhost:8080/stats"
REFRESH_RATE = 2.0  # Seconds

# Global State
state = {
    "agents": 0,
    "queue": 0,
    "rate": 0.0,
    "status": "disconnected"
}

def create_image(color):
    """Generates a simple colored dot icon on the fly."""
    width = 64
    height = 64
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    dc = ImageDraw.Draw(image)
    dc.ellipse((8, 8, 56, 56), fill=color)
    return image

def get_stats():
    """Polls the swarm controller."""
    try:
        response = requests.get(CONTROLLER_URL, timeout=1)
        data = response.json()
        
        state["agents"] = data.get("agents_online", 0)
        state["queue"] = data.get("queue_len", 0)
        state["rate"] = data.get("rate_60s", 0.0)
        
        # Determine Status Color
        if state["queue"] > 0 or state["rate"] > 1.0:
            state["status"] = "working" # Blue
        else:
            state["status"] = "idle"    # Green
            
    except:
        state["status"] = "error"       # Red

def update_loop(icon):
    """Background thread to keep stats fresh."""
    icon.visible = True
    while icon.visible:
        get_stats()
        
        # Update Icon Color based on status
        if state["status"] == "working":
            icon.icon = create_image("cyan")
        elif state["status"] == "idle":
            icon.icon = create_image("lime")
        else:
            icon.icon = create_image("red")
            
        # Update Tooltip (Hover text)
        icon.title = f"Neurofabric: {state['agents']} Agents | Q: {state['queue']}"
        
        time.sleep(REFRESH_RATE)

def on_exit(icon, item):
    icon.stop()

# Build the Menu
menu = (
    item(lambda text: f"Agents Online: {state['agents']}", lambda i, t: None),
    item(lambda text: f"Queue Depth:   {state['queue']}", lambda i, t: None),
    item(lambda text: f"Speed (1m):    {state['rate']:.1f} t/s", lambda i, t: None),
    pystray.Menu.SEPARATOR,
    item('Exit', on_exit)
)

# Start the System Tray App
icon = pystray.Icon("Neurofabric", create_image("red"), "Initializing...", menu)
threading.Thread(target=update_loop, args=(icon,), daemon=True).start()
icon.run()
