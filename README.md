# Smart Bin Monitoring System

This project consists of two main components:
1. An Arduino-based IoT system that collects bin fill level data from multiple bins and sends it to ThingSpeak
2. A Streamlit web application that visualizes the data in real-time

## Arduino Setup (Receiver)

The receiver code (`Reciver Code.c`) is designed to run on a NodeMCU ESP8266 with:
- LoRa module for receiving data from multiple bin sensors
- WiFi for ThingSpeak data transmission
- Optional GSM module for SMS alerts

### Hardware Requirements

- NodeMCU ESP8266 microcontroller
- LoRa module (e.g., SX1278)
- Optional GSM module (e.g., SIM800L) for SMS alerts
- Power supply

### Configuration

Before uploading the code to your ESP8266 board:

1. Replace `YOUR_WIFI_SSID` and `YOUR_WIFI_PASSWORD` with your WiFi credentials
2. Replace `YOUR_THINGSPEAK_API_KEY` with your ThingSpeak write API key
3. Adjust the phone number for SMS alerts (if using GSM module)

### Multi-Bin Support

The system supports monitoring up to 10 different bins simultaneously:
- Each bin is identified by a unique Bin ID
- Data from each bin is stored separately
- The system tracks fill levels independently for each bin
- Alerts are sent only for bins that exceed the threshold

## Streamlit Application

The Streamlit app (`smart_bin_monitoring.py`) provides a real-time dashboard for monitoring multiple bin fill levels and locations.

### Features

- Real-time monitoring of all bin fill levels
- Interactive map showing all bin locations with color-coded status
- Individual bin analysis with historical data
- Route planning for efficient bin collection (optimized waypoints for full bins)
- Support for multiple bins with individual tracking

### Setup

1. Install Python dependencies:
   ```
   CMD: pip install -r requirements.txt
   ```

2. Configure ThingSpeak connection:
   - Open `smart_bin_monitoring.py`
   - Replace `YOUR_CHANNEL_ID` with your ThingSpeak channel ID
   - Replace `YOUR_READ_API_KEY` with your ThingSpeak read API key

3. Run the application:
   ```
   CMD: streamlit run smart_bin_monitoring.py
   ```

### ThingSpeak Channel Setup

For this system to work, set up your ThingSpeak channel with the following fields:
- Field 1: Bin ID
- Field 2: Fill Level (%)
- Field 3: Latitude
- Field 4: Longitude

## System Architecture

```
                                ┌───────────────┐
                                │ Bin 1 Sensor  │
                                │ (Transmitter) │
                                └───────┬───────┘
                                        │
                                        │ LoRa
                                        │ 433MHz
                                        ▼
┌───────────────┐   LoRa   ┌───────────────┐    WiFi      ┌───────────────┐
│ Bin 2 Sensor  │ ─────────┤ Receiver      │ ─────────────┤ ThingSpeak    │
│ (Transmitter) │  433MHz  │ (ESP8266)     │   HTTP       │ Cloud         │
└───────────────┘          └───────────────┘              └───────────────┘
                                 │                               │
                                 │ SMS Alert                     │ API
                                 ▼ (optional)                    │
                          ┌───────────────┐                      │
                          │ Phone         │                      │
                          └───────────────┘                      │
                                                                 │
                                                                 ▼
                                                        ┌───────────────┐
                                                        │ Streamlit     │
                                                        │ Dashboard     │
                                                        └───────────────┘
```



