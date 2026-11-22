/**
 * SmartPepper ESP32 Farm Node v3.0
 * ==================================
 * Platform: Arduino framework on ESP32
 * Hardware: ESP32 DevKit V1 + OV2640 Camera + DHT22/BME280
 * Fungsi: Capture foto daun lada + sensor lingkungan
 *          Kirim ke https://smartpepper.andrian-syah.my.id/api/capture
 */

#include <WiFi.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "time.h"

// ============================================================
// KONFIGURASI USER
// ============================================================
const char* WIFI_SSID    = "ITEBA-Lab";     // Ganti nama WiFi
const char* WIFI_PASS    = "iteba2026!";    // Ganti password WiFi

// Server SmartPepper API
#define SERVER_HOST   "smartpepper.andrian-syah.my.id"
#define SERVER_PORT   443
#define API_BASE      "/api/esp32-capture"
#define NODE_NAME     "esp32-farm-1"

// Timing
#define CAPTURE_INTERVAL_SEC  60
#define TELEMETRY_INTERVAL_MS 30000

// ============================================================
// OBJEK GLOBAL
// ============================================================
WiFiClientSecure espWiFi;  // HTTPS client
PubSubClient mqttClient(espWiFi);

uint32_t g_captureCount = 0;
unsigned long g_lastCaptureTime = 0;
unsigned long g_lastTelemetryTime = 0;
unsigned long g_lastReconnectAttempt = 0;

bool wifiConnected = false;
int32_t lastRSSI = -100;

// LED indikator
#define LED_WIFI_PIN  2
#define LED_DATA_PIN  4

// ============================================================
// HELPERS
// ============================================================

void logMsg(const char* tag, const String& msg) {
  Serial.printf("[SMARTPEPPER][%s] %s\n", tag, msg.c_str());
}

String getTimestampISO() {
  struct tm ts;
  time_t now = time(nullptr);
  localtime_r(&now, &ts);
  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &ts);
  return String(buf);
}

uint32_t getTimeEpoch() {
  return (uint32_t)time(nullptr);
}

// ============================================================
// WIFI + HTTPS SETUP
// ============================================================

bool connectToWiFi() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  
  logMsg("WIFI", "Connecting to '" WIFI_SSID "'...");
  
  unsigned long startTime = millis();
  int attempts = 0;
  
  while (WiFi.status() != WL_CONNECTED && attempts < 10) {
    delay(1000);
    Serial.print(".");
    attempts++;
    if (millis() - startTime > 15000) break;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    logMsg("WIFI", "Connected!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    lastRSSI = WiFi.RSSI();
    Serial.print("RSSI: "); Serial.print(lastRSSI); Serial.println(" dBm");
    digitalWrite(LED_WIFI_PIN, HIGH);
    return true;
  }
  
  logMsg("WIFI", "FAILED!");
  digitalWrite(LED_WIFI_PIN, LOW);
  return false;
}

void verifyWiFiStatus() {
  if (WiFi.status() != WL_CONNECTED) {
    logMsg("WIFI", "Lost connection");
    connectToWiFi();
  }
}

// ============================================================
// CAMERA
// ============================================================

bool initCamera() {
  logMsg("CAM", "Init OV2640...");
  
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
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;    // 320x240
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    logMsg("CAM", "Failed! Error: 0x" + String(err, HEX));
    return false;
  }
  
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 2);
    s->set_saturation(s, 0);
    s->set_gainceiling(s, GAINCEILING_8X);
    logMsg("CAM", "OK");
  }
  return true;
}

camera_fb_t* captureFrame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    logMsg("CAM", "Capture FAILED");
    return nullptr;
  }
  g_captureCount++;
  logMsg("CAM", "Frame #" + String(g_captureCount) + 
         ": " + String(fb->width) + "x" + String(fb->height) + 
         ", " + String(fb->len) + " bytes");
  return fb;
}

// ============================================================
// SENSOR DATA (DHT22/I2C)
// ============================================================

float readTemperature() {
  static float t = 32.0f;
  // TODO: Replace with real BME280/DHT22 I2C read
  t += random(-2, 3) / 100.0f;
  return constrain(t, 20.0f, 45.0f);
}

float readHumidity() {
  static float h = 85.0f;
  h += random(-3, 4) / 100.0f;
  return constrain(h, 40.0f, 100.0f);
}

uint16_t readSoilMoisture() {
  // TODO: ADC pin untuk sensor kelembaban tanah
  return 65; // dummy
}

// ============================================================
// UPLOAD FOTO KE SERVER SMARTPEPPER
// ============================================================

// Multipart form boundary
#define FORM_BOUNDARY "SmartPepperESP32---"

struct HTTPResponse {
  bool success;
  uint16_t statusCode;
  String body;
};

HTTPResponse postImageAsMultipart(camera_fb_t* fb, float tempC, float humPct) {
  HTTPResponse resp;
  resp.success = false;
  resp.statusCode = 0;
  
  // Build multipart/form-data payload
  // Format: --boundary\r\nContent-Disposition: form-data; name="field"\r\n\r\nvalue\r\n--boundary\r\nContent-Disposition: form-data; name="file"; filename="capture.jpg"\r\nContent-Type: image/jpeg\r\n\r\n<BINARY IMAGE DATA>\r\n--boundary--\r\n
  
  // Hitung ukuran total payload
  size_t headerSize = 0;
  const char* fieldHeaders[] = {
    "--" FORM_BOUNDARY "\r\nContent-Disposition: form-data; name=\"node\"\r\n\r\n",
    NODE_NAME "\r\n--" FORM_BOUNDARY "\r\nContent-Disposition: form-data; name=\"timestamp\"\r\n\r\n",
  };
  
  String tsStr = String(getTimeEpoch());
  DateTime isoTs = DateTime(getTimestampISO());
  
  // Perhitungan kasar: header + metadata fields + file header + image data + closing
  size_t totalEstimate = 200 + 256 + fb->len + 50;
  
  // Setup HTTPS connection
  espWiFi.setInsecure();  // Skip cert verification (Cloudflare SSL still works)
  
  if (!espWiFi.connect(SERVER_HOST, SERVER_PORT)) {
    logMsg("HTTP", "Cannot connect to " SERVER_HOST);
    return resp;
  }
  
  if (!espWiFi.connected()) {
    logMsg("HTTP", "Connection failed");
    espWiFi.stop();
    return resp;
  }
  
  logMsg("HTTP", "Connected to " SERVER_HOST ":" + String(SERVER_PORT));
  
  // Construct and send request
  String requestBody;
  requestBody.reserve(fb->len + 512);
  
  // Metadata fields
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"node\"\r\n\r\n";
  requestBody += NODE_NAME "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"timestamp\"\r\n\r\n";
  requestBody += tsStr + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"datetime_iso\"\r\n\r\n";
  requestBody += getTimestampISO() + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"wifi_rssi\"\r\n\r\n";
  requestBody += String(lastRSSI) + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"heap_free\"\r\n\r\n";
  requestBody += String(ESP.getFreeHeap()) + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"captures_total\"\r\n\r\n";
  requestBody += String(g_captureCount) + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"temperature_c\"\r\n\r\n";
  requestBody += String(tempC, 1) + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"humidity_pct\"\r\n\r\n";
  requestBody += String(humPct, 1) + "\r\n";
  
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"soil_moisture\"\r\n\r\n";
  requestBody += String(readSoilMoisture()) + "\r\n";
  
  // Image file part
  requestBody += "--" FORM_BOUNDARY "\r\n";
  requestBody += "Content-Disposition: form-data; name=\"image\"; filename=\"capture_" + String(g_captureCount) + ".jpg\"\r\n";
  requestBody += "Content-Type: image/jpeg\r\n\r\n";
  
  size_t written = espWiFi.print(requestBody);
  logMsg("HTTP", "Headers sent: " + String(written) + " bytes");
  
  // Send binary image data directly
  written = espWiFi.write(fb->buf, fb->len);
  logMsg("HTTP", "Image data sent: " + String(written) + " bytes");
  
  // Closing boundary
  espWiFi.print("\r\n--" FORM_BOUNDARY "--\r\n");
  
  // Close TCP connection
  delay(100);  // Wait for response
  espWiFi.flush();
  
  // Read response headers
  String responseLine = "";
  while (espWiFi.available()) {
    responseLine += (char)espWiFi.read();
    if (responseLine.indexOf("\r\n\r\n") >= 0) break;
  }
  
  // Try to read JSON response body
  String responseBody = "";
  while (espWiFi.available()) {
    responseBody += (char)espWiFi.read();
  }
  
  // Parse status code from HTTP version line
  uint16_t statusCode = 400;  // Default error
  if (responseLine.indexOf("200") >= 0 || responseBody.indexOf("success") >= 0) {
    statusCode = 200;
    resp.success = true;
  } else if (responseLine.indexOf("500") >= 0) {
    statusCode = 500;
  } else if (responseLine.indexOf("404") >= 0) {
    statusCode = 404;
  } else if (responseLine.indexOf("422") >= 0) {
    statusCode = 422;
  }
  
  resp.statusCode = statusCode;
  resp.body = responseBody;
  
  logMsg("HTTP", "Response: " + String(statusCode) + 
         " (" + String(resp.body.length()) + " bytes)");
  
  // Cleanup
  espWiFi.stop();
  esp_camera_fb_return(fb);  // Release frame buffer
  
  return resp;
}

// ============================================================
// SENSOR TELEMETRY VIA HTTP
// ============================================================

HTTPResponse publishSensorTelemetry(float tempC, float humPct) {
  HTTPResponse resp;
  resp.success = false;
  
  StaticJsonDocument<512> doc;
  doc["node"] = NODE_NAME;
  doc["type"] = "telemetry";
  doc["timestamp"] = getTimeEpoch();
  doc["temperature_c"] = tempC;
  doc["humidity_pct"] = humPct;
  doc["captures_total"] = g_captureCount;
  doc["wifi_rssi"] = lastRSSI;
  doc["heap_free"] = ESP.getFreeHeap();
  doc["uptime_hours"] = millis() / 3600000.0;
  
  char jsonBuf[512];
  serializeJson(doc, jsonBuf);
  
  espWiFi.setInsecure();
  if (!espWiFi.connect(SERVER_HOST, SERVER_PORT)) {
    resp.success = false;
    return resp;
  }
  
  espHttpPutRequest("/api/esp32-telemetry", jsonBuf, strlen(jsonBuf), "application/json");
  
  resp.success = true;
  espWiFi.stop();
  return resp;
}

// Helper: simple HTTP PUT for telemetry
void espHttpPutRequest(const char* path, const char* body, size_t bodyLen, const char* contentType) {
  String request = "PUT ";
  request += path;
  request += " HTTP/1.1\r\n";
  request += "Host: " SERVER_HOST "\r\n";
  request += "Content-Type: " + String(contentType) + "\r\n";
  request += "Content-Length: " + String(bodyLen) + "\r\n";
  request += "Connection: close\r\n\r\n";
  request += String(body);
  
  espWiFi.print(request);
  delay(200);
  espWiFi.flush();
  espWiFi.stop();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  verifyWiFiStatus();
  
  unsigned long now = millis();
  
  // ===== PERIODIC CAPTURE & UPLOAD =====
  if (now - g_lastCaptureTime > ((uint32_t)CAPTURE_INTERVAL_SEC * 1000UL)) {
    g_lastCaptureTime = now;
    
    logMsg("LOOP", "=== Cycle #" + String(g_captureCount + 1) + " ===");
    digitalWrite(LED_DATA_PIN, HIGH);  // Indicator ON
    
    // Take photo
    camera_fb_t* fb = captureFrame();
    if (!fb) {
      logMsg("ERROR", "No frame captured");
      digitalWrite(LED_DATA_PIN, LOW);
      delay(CAPTURE_INTERVAL_SEC * 1000);
      return;
    }
    
    // Read sensors
    float tempC = readTemperature();
    float humPct = readHumidity();
    
    // Upload to server via multipart POST
    logMsg("HTTP", "Uploading to " SERVER_HOST "...");
    HTTPResponse result = postImageAsMultipart(fb, tempC, humPct);
    
    if (result.success) {
      logMsg("OK", "Upload successful! Server returned: " + result.body.substring(0, 50));
    } else {
      logMsg("FAIL", "Upload failed: HTTP " + String(result.statusCode));
    }
    
    digitalWrite(LED_DATA_PIN, LOW);
    yield();
  }
  
  // ===== PERIODIC SENSOR TELEMETRY =====
  if (now - g_lastTelemetryTime > TELEMETRY_INTERVAL_MS) {
    g_lastTelemetryTime = now;
    
    float tempC = readTemperature();
    float humPct = readHumidity();
    
    HTTPResponse result = publishSensorTelemetry(tempC, humPct);
    
    if (result.success) {
      logMsg("TELEMETRY", "Sent successfully");
    } else {
      logMsg("TELEMETRY", "Send failed");
    }
  }
  
  yield();
  delay(50);
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(500);
  
  Serial.println("\n====================================");
  Serial.println("  SMARTPEPPER ESP32 FARM NODE v3.0");
  Serial.println("  Target: " SERVER_HOST);
  Serial.println("====================================");
  
  // Initialize indicators
  pinMode(LED_WIFI_PIN, OUTPUT);
  pinMode(LED_DATA_PIN, OUTPUT);
  digitalWrite(LED_WIFI_PIN, LOW);
  digitalWrite(LED_DATA_PIN, LOW);
  
  // Init camera
  if (!initCamera()) {
    logMsg("FATAL", "Camera init failed");
  }
  
  // Connect WiFi
  connectToWiFi();
  
  logMsg("SYSTEM", "Ready. Starting main loop.");
}
