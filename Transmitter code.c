#include <LoRa.h>

const int trigPin = 4;   // D2
const int echoPin = 2;   // D4

#define LORA_SS    15    // D8
#define LORA_RST   5     // D1
#define LORA_DIO0  16    // D0

const String dustbinID = "DB01";   //Replace with actual bin ID
const String latitude  = "17.3850";   // Replace with actual latitude
const String longitude = "78.4867";   // Replace with actual longitude
const float binHeight  = 30.0; // Replace with actual bin height in cm

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(433E6)) while (1);
}

float getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000);
  if (duration == 0) return -1;
  return (duration * 0.0343) / 2.0;
}

int calculateFillLevel(float distance) {
  if (distance < 0) return -1;
  int percent = ((binHeight - distance) / binHeight) * 100;
  return constrain(percent, 0, 100);
}

void loop() {
  float d1 = getDistance(); delay(50);
  float d2 = getDistance(); delay(50);
  float d3 = getDistance();

  float avg = 0;
  int valid = 0;

  if (d1 > 0) { avg += d1; valid++; }
  if (d2 > 0) { avg += d2; valid++; }
  if (d3 > 0) { avg += d3; valid++; }

  if (valid > 0) {
    avg /= valid;
    int fillLevel = calculateFillLevel(avg);

    String message = "ID:" + dustbinID +
                     ",FILL:" + String(fillLevel) +
                     ",LOC:" + latitude + "," + longitude;
    LoRa.beginPacket();
    LoRa.print(message);
    LoRa.endPacket();
  }

  delay(10000);
}
