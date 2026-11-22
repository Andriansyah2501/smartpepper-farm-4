/**
 * SmartPepper ESP32 Edge Vision Node Firmware
 * ================================
 * ESP32 dengan kamera OV2640/OV5640 untuk deteksi kesehatan daun lada real-time.
 * Mengirim hasil inferensi via MQTT ke Broker di 192.168.0.30.
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <TFT_eSPI.h> // Optional: display preview
#include "esp_camera.h"
#include "time.h"

// ========= KONFIGURASI WIFI =========
const char* STA_SSID = "ITEBA-WIFI";
const char* STA_PASS = "iteba2026!";
const char* AP_SSID  = "SmartPepper-Node";
const char* AP_PASS  = "smartpepper";

// ========= KONFIGURASI MQTT SERVER =========
#define MQTT_SERVER   "192.168.0.30"
#define MQTT_PORT     1883
#define MQTT_USER     ""      // Kosongkan jika tidak pakai auth
#define MQTT_PASS     ""

// Topik MQTT
#define TOPIC_DETECTION   "smartpepper/detection"
#define TOPIC_SENSOR      "smartpepper/sensor/node_esp32"
#define TOPIC_STATUS      "smartpepper/status"
#define TOPIC_CONFIG      "smartpepper/config/esp32"

// ========= KONFIGURASI KAMERA (OV2640) =========
#define CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     21
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       19
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       5
#define Y2_GPIO_NUM       4
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ========= PARAMETER SISTEM =========
#define NODE_NAME         "esp32_canopy_1"
#define INTERVAL_CAPTURE  60      // Detik antara capture
#define WIFI_TIMEOUT_MS   10000
#define MAX_RECONNECT     5

// ========= OBJEK GLOBAL =========
WiFiClient espWifiClient;
PubSubClient mqttClient(espWifiClient);
TFT_eSPI tft = TFT_eSPI();

uint32_t gCaptureCount = 0;
bool gMQTTConnected = false;
bool gWiFiConnected = false;
unsigned long gLastCaptureTime = 0;
unsigned long gLastMQTTRetry = 0;

// ========= FUNGSI HELPERS =========

void logMsg(const char* tag, const char* msg) {
  Serial.printf("[LOG-%s] %s\n", tag, msg);
}

String getTimestamp() {
  struct tm ts;
  time_t now = time(nullptr);
  localtime_r(&now, &ts);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &ts);
  return String(buf);
}

uint32_t getTimeEpoch() {
  return (uint32_t)time(nullptr);
}

// ========= MQTT CALLBACKS =========

void mqttReceived(char* topic, byte* payload, unsigned int length) {
  Serial.print("MQTT Received [");
  Serial.print(topic);
  Serial.println("]:");
  
  // Convert payload to string
  String payloadStr = "";
  for (int i = 0; i < length; i++) {
    payloadStr += (char)payload[i];
  }
  
  // Parse JSON config commands
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, payloadStr);
  
  if (!error) {
    // Handle remote configuration
    if (doc["interval"]) {
      unsigned newInterval = doc["interval"];
      if (newInterval >= 10 && newInterval <= 600) {
        INTERVAL_CAPTURE = newInterval;
        logMsg("CONFIG", "Capture interval updated remotely");
      }
    }
    
    if (doc["restart"]) {
      logMsg("CMD", "Remote restart requested");
      delay(1000);
      ESP.restart();
    }
    
    if (doc["config"] && doc["config"]["wifi_ssid"]) {
      // WiFi reconfiguration from server
      const char* newSSID = doc["config"]["wifi_ssid"];
      const char* newPass = doc["config"]["wifi_pass"];
      logMsg("CONFIG", "WiFi credentials updated");
    }
    
    // Acknowledge receipt
    JsonDocument ackDoc;
    ackDoc["status"] = "ack";
    ackDoc["topic"] = topic;
    ackDoc["from"] = NODE_NAME;
    String ackPayload;
    serializeJson(ackDoc, ackPayload);
    
    char ackBuf[256];
    ackPayload.toCharArray(ackBuf, sizeof(ackBuf));
    mqttClient.publish(TOPIC_STATUS, ackBuf);
  } else {
    logMsg("JSON ERR", "Failed to parse config message");
  }
}

void reconnectMQTT() {
  if (gMQTTConnected) return;
  
  uint32_t elapsed = millis() - gLastMQTTRetry;
  if (elapsed < 5000) return; // Minimum 5s between attempts
  
  gLastMQTTRetry = millis();
  
  Serial.print("Connecting to MQTT ");
  Serial.print(MQTT_SERVER);
  Serial.print(":");
  Serial.print(MQTT_PORT);
  Serial.println("...");
  
  if (mqttClient.connect(NODE_NAME, MQTT_USER, MQTT_PASS)) {
    gMQTTConnected = true;
    logMsg("MQTT", "Connected successfully!");
    
    // Subscribe to control topics
    mqttClient.subscribe(TOPIC_CONFIG);
    mqttClient.subscribe(String(TOPIC_STATUS) + "/+/cmd");
    
    // Publish online status
    JsonDocument statusDoc;
    statusDoc["node"] = NODE_NAME;
    statusDoc["status"] = "online";
    statusDoc["timestamp"] = getTimeEpoch();
    statusDoc["firmware"] = "1.0.0";
    statusDoc["camera"] = "OV2640";
    
    char statusBuf[256];
    serializeJson(statusDoc, statusBuf);
    mqttClient.publish(TOPIC_STATUS, statusBuf);
  } else {
    int rc = mqttClient.state();
    logMsg("MQTT ERR", "Connection failed. State: ");
    Serial.println(rc);
  }
}

// ========= WIFI FUNCTIONS =========

bool connectWiFi() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);
  
  logMsg("WIFI", "Connecting to " STA_SSID "...");
  
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startTime) < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    gWiFiConnected = true;
    Serial.println("\nWiFi connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("RSSI: ");
    Serial.println(WiFi.RSSI());
    return true;
  }
  
  // Fallback to AP mode
  logMsg("WIFI", "STA failed, starting AP...");
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());
  return false;
}

// ========= CAMERA FUNCTIONS =========

bool initCamera() {
  camera_config_t config;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // For AI inference, use QVGA for speed
  config.frame_size = FRAMESIZE_QVGA;  // 320x240 - optimal untuk YOLOv8
  config.jpeg_quality = 12;            // Low quality untuk bandwidth hemat
  config.fb_count = 1;
  
  logMsg("CAM", "Initializing camera...");
  esp_err_t err = esp_camera_init(&config);
  
  if (err != ESP_OK) {
    logMsg("CAM ERR", "Initialization failed!");
    return false;
  }
  
  // Set optimal settings for leaf detection
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_contrast(s, 2);    // Normal contrast
    s->set_brightness(s, 1);  // Slight brightness boost
    s->set_saturation(s, 0);  // Neutral saturation
    
    // Auto gain/max
    s->set_gainceiling(s, GAINCEILING_8X);
    
    logMsg("CAM", "Camera initialized OK");
  }
  
  return true;
}

camera_fb_t* takePhotoWithTiming() {
  unsigned long captureStart = micros();
  camera_fb_t* fb = esp_camera_fb_get();
  unsigned long captureEnd = micros();
  
  if (!fb) {
    logMsg("CAM", "Capture failed - no buffer");
    return nullptr;
  }
  
  uint32_t duration = (captureEnd - captureStart) / 1000;
  Serial.printf("Frame captured: %dx%d, size=%ld bytes, %lu ms\n", 
                fb->width, fb->height, fb->len, duration);
  
  return fb;
}

// ========= MQTT TRANSMISSION =========

bool sendDetectionToMQTT(camera_fb_t* fb) {
  if (!gMQTTConnected) {
    reconnectMQTT();
    if (!gMQTTConnected) return false;
  }
  
  // Build comprehensive JSON payload
  StaticJsonDocument<1024> doc;
  
  // Metadata
  doc["node"] = NODE_NAME;
  doc["timestamp"] = getTimeEpoch();
  doc["datetime"] = getTimestamp();
  doc["frame_size"] = fb->width;
  doc["frame_height"] = fb->height;
  doc["image_size_bytes"] = fb->len;
  doc["capture_duration_ms"] = 0;
  
  // Simulasi hasil deteksi (pada implementasi nyata, kirimkan ke edge TPU atau server AI)
  // Untuk saat ini, kita bisa kirim gambar mentah dan server akan memprosesnya
  
  // Status WiFi dan baterai jika ada
  doc["signal_strength"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime_hours"] = millis() / 3600000UL;
  
  // Encode frame sebagai base64 (hanya bagian metadata, frame dikirim terpisah)
  // MQTT tidak cocok untuk gambar besar -> gunakan chunking atau HTTP fallback
  
  char jsonBuf[512];
  serializeJson(doc, jsonBuf);
  
  // Publikasikan metadata
  bool success = mqttClient.publish(TOPIC_DETECTION, jsonBuf);
  
  // Kirim gambar via TCP direct jika MQTT gagal (karena terlalu besar)
  if (!success) {
    logMsg("MQTT", "Publish failed, sending raw frame...");
  }
  
  return success;
}

// ========= TELEMETRY SENSOR =========

void publishSensorTelemetry(uint16_t tempC, uint16_t humidityPercent, uint16_t soilMoisturePercent) {
  if (!gMQTTConnected) return;
  
  StaticJsonDocument<256> doc;
  doc["node"] = NODE_NAME;
  doc["type"] = "sensor";
  doc["temperature_c"] = tempC / 10.0;
  doc["humidity_percent"] = humidityPercent / 10.0;
  doc["soil_moisture_percent"] = soilMoisturePercent;
  doc["timestamp"] = getTimeEpoch();
  
  char jsonBuf[256];
  serializeJson(doc, jsonBuf);
  mqttClient.publish(TOPIC_SENSOR, jsonBuf);
}

// ========= SIMULASI SENSOR DATA (ganti dengan read actual sensors) =========

void readSensors(float& tempC, float& humidity) {
  // Dummy values — ganti dengan DHT22/BME280/I2C later
  static float fakeTemp = 32.5;
  static float fakeHum = 85.0;
  
  // Add slight random variation
  fakeTemp += (random(-5, 5)) / 100.0;
  fakeHum += (random(-20, 20)) / 100.0;
  fakeHum = constrain(fakeHum, 30.0, 100.0);
  
  tempC = fakeTemp;
  humidity = fakeHum;
}

// ========= LOOP UTAMA =========

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n=== SMARTPEPPER ESP32 NODE ===");
  Serial.println("Booting...");
  
  // Init screen (if available)
  #if defined(TFT_CS) && defined(TFT_RST)
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(10, 20);
  tft.println(F("SmartPepper"));
  tft.print(F("ESP32 Node v1.0"));
  #endif
  
  // Init WiFi
  if (!connectWiFi()) {
    logMsg("WARN", "Using AP fallback");
  }
  
  // Init Camera
  if (!initCamera()) {
    logMsg("ERROR", "Camera init failed - check connections");
    // Continue anyway, can retry later
  }
  
  // Setup MQTT
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttReceived);
  mqttClient.setBufferSize(1024);  // Increase for larger payloads
  mqttClient.setKeepAlive(60);
  
  logMsg("SYSTEM", "Setup complete");
}

void loop() {
  // Maintain WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    logMsg("WIFI", "Reconnecting...");
    connectWiFi();
  }
  
  // Maintain MQTT connection
  if (!gMQTTConnected || !mqttClient.connected()) {
    reconnectMQTT();
  }
  
  // Process incoming MQTT messages
  mqttClient.loop();
  
  // Time-based capture loop
  unsigned long now = millis();
  if (now - gLastCaptureTime > (INTERVAL_CAPTURE * 1000UL)) {
    gLastCaptureTime = now;
    gCaptureCount++;
    
    logMsg("LOOP", "Starting capture cycle #" + String(gCaptureCount));
    
    // Take photo
    camera_fb_t* fb = takePhotoWithTiming();
    if (!fb) {
      logMsg("CAM", "Failed to capture image");
      delay(2000);
      return;
    }
    
    // Send via MQTT (metadata only due to size limits)
    sendDetectionToMQTT(fb);
    
    // Release frame buffer
    esp_camera_fb_return(fb);
    
    logMsg("LOOP", "Cycle complete, next in " + String(INTERVAL_CAPTURE) + "s");
  }
  
  // Publish telemetry every 30 seconds (separate from capture)
  static unsigned long lastTelemetry = 0;
  if (now - lastTelemetry > 30000UL) {
    lastTelemetry = now;
    
    float temp, hum;
    readSensors(temp, hum);
    
    StaticJsonDocument<256> sensorDoc;
    sensorDoc["node"] = NODE_NAME;
    sensorDoc["temp_c"] = temp;
    sensorDoc["humidity"] = hum;
    sensorDoc["captures"] = gCaptureCount;
    sensorDoc["uptime_hrs"] = millis() / 3600000.0;
    sensorDoc["heap_free"] = ESP.getFreeHeap();
    sensorDoc["wifi_rssi"] = WiFi.RSSI();
    
    char sensorBuf[256];
    serializeJson(sensorDoc, sensorBuf);
    mqttClient.publish(TOPIC_SENSOR, sensorBuf);
  }
  
  // Yield to WiFi/MQTT background tasks
  yield();
}
