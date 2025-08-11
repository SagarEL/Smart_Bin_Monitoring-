#include <SPI.h>
#include <LoRa.h>
#include <SoftwareSerial.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

#define LORA_SS   15  // D8
#define LORA_RST  5   // D1
#define LORA_DIO0 16  // D0

SoftwareSerial gsmSerial(4, 2);  // RX, TX

const char* ssid     = "YOUR_ACTUAL_WIFI_SSID";
const char* password = "YOUR_ACTUAL_WIFI_PASSWORD";

const String thingspeakAPIKey = "4XASJL05RLSEPSEJ";
const String thingspeakServer = "api.thingspeak.com";
unsigned long lastThingspeakUpdate = 0;
const unsigned long THINGSPEAK_INTERVAL = 20000;

#define MAX_BINS 10
#define BUFFER_SIZE 50
#define AVG_INTERVAL 15000

struct BinData {
  String binID;
  String latitude;
  String longitude;
  int fillBuffer[BUFFER_SIZE];
  int bufferIndex;
  int lastAvgFill;
  bool messageSent;
};

BinData bins[MAX_BINS];
int numBins = 0;

bool wifiConnected = false;
bool gsmConnected = false;
unsigned long lastAvgTime = 0;

void setup() {
  Serial.begin(9600);
  gsmSerial.begin(9600);

  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(433E6)) while (true);

  setupWiFi();
  
  gsmConnected = checkGSMConnection();
  if (gsmConnected) {
    gsmSerial.println("AT+CMGF=1");
    delay(500);
  }

  lastAvgTime = millis();
  lastThingspeakUpdate = millis();

  for (int i = 0; i < MAX_BINS; i++) {
    bins[i].binID = "";
    bins[i].bufferIndex = 0;
    bins[i].lastAvgFill = 0;
    bins[i].messageSent = false;
  }
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) setupWiFi();

  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String received = LoRa.readString();
    String binID = getValue(received, ',', 0);
    String fillStr = getValue(received, ',', 1);
    String latStr = getValue(received, ',', 2);
    String lonStr = getValue(received, ',', 3);

    binID = binID.substring(binID.indexOf(':') + 1);
    fillStr = fillStr.substring(fillStr.indexOf(':') + 1);
    latStr = latStr.substring(latStr.indexOf(':') + 1);
    lonStr = lonStr.substring(lonStr.indexOf(':') + 1);

    int fill = fillStr.toInt();
    int binIndex = findBinIndex(binID);

    if (binIndex == -1 && numBins < MAX_BINS) {
      binIndex = numBins++;
      bins[binIndex].binID = binID;
    }

    bins[binIndex].latitude = latStr;
    bins[binIndex].longitude = lonStr;
    if (bins[binIndex].bufferIndex < BUFFER_SIZE) {
      bins[binIndex].fillBuffer[bins[binIndex].bufferIndex++] = fill;
    }
  }

  if (millis() - lastAvgTime >= AVG_INTERVAL) {
    lastAvgTime = millis();
    for (int i = 0; i < numBins; i++) {
      if (bins[i].bufferIndex > 0) {
        int sum = 0;
        for (int j = 0; j < bins[i].bufferIndex; j++) sum += bins[i].fillBuffer[j];
        bins[i].lastAvgFill = sum / bins[i].bufferIndex;

        if (bins[i].lastAvgFill > 85 && !bins[i].messageSent) {
          String msg = "🗑️ Bin Alert!\nBin ID: " + bins[i].binID +
                       "\nAvg Fill: " + String(bins[i].lastAvgFill) + "%" +
                       "\nLocation: https://maps.google.com/?q=" + bins[i].latitude + "," + bins[i].longitude;
          if (gsmConnected) {
            sendSMS(msg);
            bins[i].messageSent = true;
          }
        }
        if (bins[i].lastAvgFill < 50) bins[i].messageSent = false;
        bins[i].bufferIndex = 0;
      }
    }
  }

  if (millis() - lastThingspeakUpdate >= THINGSPEAK_INTERVAL) {
    lastThingspeakUpdate = millis();
    if (wifiConnected && numBins > 0) {
      for (int i = 0; i < numBins; i++) {
        if (bins[i].binID != "") {
          sendToThingSpeak(bins[i].binID, bins[i].lastAvgFill, bins[i].latitude, bins[i].longitude);
          delay(16000);
        }
      }
    } else if (!wifiConnected) setupWiFi();
  }
}


void setupWiFi() {
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    attempts++;
  }
  wifiConnected = (WiFi.status() == WL_CONNECTED);
}

void sendToThingSpeak(String binID, int fillLevel, String lat, String lon) {
  if (WiFi.status() != WL_CONNECTED) return;
  WiFiClient client;
  HTTPClient http;
  String url = "http://" + thingspeakServer + "/update?api_key=" + thingspeakAPIKey +
               "&field1=" + urlEncode(binID) +
               "&field2=" + String(fillLevel) +
               "&field3=" + lat +
               "&field4=" + lon;
  http.begin(client, url);
  http.GET();
  http.end();
}

void sendSMS(String message) {
  gsmSerial.println("AT+CMGS=\"+91XXXXXXXXXX\"");
  delay(500);
  gsmSerial.print(message);
  delay(500);
  gsmSerial.write(26);
  delay(3000);
}

bool checkGSMConnection() {
  gsmSerial.println("AT");
  unsigned long timeout = millis();
  while (millis() - timeout < 3000) {
    if (gsmSerial.available()) {
      String response = gsmSerial.readString();
      if (response.indexOf("OK") != -1) return true;
    }
  }
  return false;
}

String getValue(String data, char separator, int index) {
  int found = 0, start = 0;
  for (int i = 0; i <= data.length(); i++) {
    if (data.charAt(i) == separator || i == data.length()) {
      if (found == index) return data.substring(start, i);
      found++;
      start = i + 1;
    }
  }
  return "";
}

int findBinIndex(String binID) {
  for (int i = 0; i < numBins; i++) {
    if (bins[i].binID == binID) return i;
  }
  return -1;
}

String urlEncode(String str) {
  String encodedString = "";
  for (int i = 0; i < str.length(); i++) {
    char c = str.charAt(i);
    if (c == ' ') encodedString += '+';
    else if (isAlphaNumeric(c)) encodedString += c;
    else {
      char code1 = (c & 0xf) + '0';
      if ((c & 0xf) > 9) code1 = (c & 0xf) - 10 + 'A';
      c = (c >> 4) & 0xf;
      char code0 = c + '0';
      if (c > 9) code0 = c - 10 + 'A';
      encodedString += '%';
      encodedString += code0;
      encodedString += code1;
    }
  }
  return encodedString;
}
