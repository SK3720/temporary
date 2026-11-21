#pragma once
// Word -> Google Translate -> LCD + speech (Cloud TTS, LINEAR16 -> MAX98357A).
//
// If your team already has a working Translate + TTS path from the speech
// levels, call it from say_word() between the camera_stop()/camera_start()
// that is already there (classic ESP32 does not have the RAM for camera +
// WiFi + TLS at once).
#include "config.h"
#include "vision.h"
#include "ui.h"
#if USE_NETWORK
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif

static void network_begin() {        // non-blocking; connects in the background
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

static bool wifi_up() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);  // retry a failed / dropped connection
  return WiFi.waitForConnectResult(10000) == WL_CONNECTED;
}

static String json_escape(const String& s) {
  String o;
  for (size_t i = 0; i < s.length(); i++) {
    if (s[i] == '"' || s[i] == '\\') o += '\\';
    o += s[i];
  }
  return o;
}

// POST a JSON body to a googleapis.com endpoint. Caller reads the response and calls http.end().
static int google_post(HTTPClient& http, WiFiClientSecure& tls, const char* url, const String& body) {
  tls.setInsecure();               // hackathon: skip cert pinning
  http.setTimeout(15000);
  if (!http.begin(tls, String(url) + "?key=" + GOOGLE_API_KEY)) return -1;
  http.addHeader("Content-Type", "application/json");
  return http.POST(body);
}

// Returns the translated text, or the input unchanged on failure / "en".
static String translate_text(const String& text) {
  if (!strcmp(TARGET_LANG, "en")) return text;
  WiFiClientSecure tls;
  HTTPClient http;
  int code = google_post(http, tls, "https://translation.googleapis.com/language/translate/v2",
                         String("{\"q\":\"") + json_escape(text) + "\",\"source\":\"en\",\"target\":\"" +
                         TARGET_LANG + "\",\"format\":\"text\"}");
  String body = code == 200 ? http.getString() : "";
  http.end();
  int k = body.indexOf("\"translatedText\"");
  if (k < 0) { Serial.printf("WARN translate HTTP %d\n", code); return text; }
  k = body.indexOf('"', body.indexOf(':', k)) + 1;
  String out;
  for (int i = k; i < (int)body.length() && body[i] != '"'; i++) {
    if (body[i] == '\\' && i + 1 < (int)body.length()) i++;
    out += body[i];
  }
  return out;
}

// Stream sink: finds "audioContent": "<base64>" and decodes it into PSRAM.
class AudioSink : public Stream {
 public:
  uint8_t* buf; size_t cap, len = 0;
  AudioSink(size_t c) : cap(c) { buf = (uint8_t*)ps_malloc(c); if (!buf) cap = 0; }
  ~AudioSink() { free(buf); }
  size_t write(uint8_t c) override {
    static const char* key = "\"audioContent\"";
    if (state == 0) {                       // searching for the key
      match = (c == key[match]) ? match + 1 : (c == key[0] ? 1 : 0);
      if (key[match] == 0) state = 1;
    } else if (state == 1) {                // skip ': "'
      if (c == '"') state = 2;
    } else if (state == 2) {                // base64 body
      if (c == '"') { state = 3; return 1; }
      int v = b64(c);
      if (v < 0) return 1;
      acc = (acc << 6) | v; bits += 6;
      if (bits >= 8) { bits -= 8; if (len < cap) buf[len++] = (acc >> bits) & 0xFF; }
    }
    return 1;
  }
  size_t write(const uint8_t* p, size_t n) override { for (size_t i = 0; i < n; i++) write(p[i]); return n; }
  int available() override { return 0; }
  int read() override { return -1; }
  int peek() override { return -1; }
  void flush() override {}
 private:
  int state = 0, match = 0, bits = 0;
  uint32_t acc = 0;
  static int b64(uint8_t c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  }
};

static void play_pcm16(const int16_t* s, size_t n) {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = TTS_SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.tx_desc_auto_clear = true;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_BCLK;
  pins.ws_io_num = I2S_LRC;
  pins.data_out_num = AMP_DOUT;
  pins.data_in_num = I2S_PIN_NO_CHANGE;
  if (i2s_driver_install(AMP_I2S_PORT, &cfg, 0, nullptr) != ESP_OK) { Serial.println("ERR i2s install"); return; }
  i2s_set_pin(AMP_I2S_PORT, &pins);
  int16_t chunk[256];
  for (size_t i = 0; i < n; i += 256) {
    size_t m = min((size_t)256, n - i), wr;
    for (size_t j = 0; j < m; j++) chunk[j] = (int16_t)(s[i + j] * TTS_VOLUME);
    i2s_write(AMP_I2S_PORT, chunk, m * 2, &wr, portMAX_DELAY);
  }
  memset(chunk, 0, sizeof chunk);
  size_t wr;
  for (int k = 0; k < 8; k++) i2s_write(AMP_I2S_PORT, chunk, sizeof chunk, &wr, portMAX_DELAY);
  i2s_driver_uninstall(AMP_I2S_PORT);
}

static bool tts_speak(const String& text) {
  WiFiClientSecure tls;
  HTTPClient http;
  int code = google_post(http, tls, "https://texttospeech.googleapis.com/v1/text:synthesize",
                         String("{\"input\":{\"text\":\"") + json_escape(text) +
                         "\"},\"voice\":{\"languageCode\":\"" + TTS_LANG_CODE +
                         "\"},\"audioConfig\":{\"audioEncoding\":\"LINEAR16\",\"sampleRateHertz\":" +
                         TTS_SAMPLE_RATE + "}}");
  if (code != 200) { Serial.printf("WARN TTS HTTP %d %s\n", code, http.getString().c_str()); http.end(); return false; }
  AudioSink sink(TTS_SAMPLE_RATE * 2 * 8);  // up to 8 s
  http.writeToStream(&sink);
  http.end();
  if (sink.len < 64) { Serial.println("WARN TTS: no audio in response"); return false; }
  // LINEAR16 comes as a WAV file: skip to the "data" chunk.
  size_t off = 12;
  while (off + 8 <= sink.len && memcmp(sink.buf + off, "data", 4) != 0)
    off += 8 + (sink.buf[off + 4] | sink.buf[off + 5] << 8 | sink.buf[off + 6] << 16 | sink.buf[off + 7] << 24);
  off = (off + 8 <= sink.len) ? off + 8 : 44;
  play_pcm16((const int16_t*)(sink.buf + off), (sink.len - off) / 2);
  return true;
}
#else
static void network_begin() {}
#endif  // USE_NETWORK

// Called once per finished word: translate, show on LCD row 2, speak.
static void say_word(const char* word) {
#if USE_NETWORK
  bool cam = g_cameraOn;
  camera_stop();                    // RAM for WiFi + TLS on classic ESP32
  if (wifi_up()) {
    String spoken = translate_text(word);
    lcd_row(2, "> %s", spoken.c_str());
    Serial.printf("SAY %s -> %s\n", word, spoken.c_str());
    if (!tts_speak(spoken)) Serial.println("WARN speech failed");
  } else {
    Serial.println("WARN WiFi down - showing word only");
    lcd_row(2, "> %s", word);
  }
  if (cam) camera_start();
#else
  lcd_row(2, "> %s", word);
  Serial.printf("SAY %s\n", word);
#endif
}
