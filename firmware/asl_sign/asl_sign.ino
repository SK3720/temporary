// ASL fingerspelling -> word -> translate -> LCD + speech, fully on one ESP32.
//
// Model + approach adapted from J. Randall, "American-Sign-language"
// (github.com/jonathanrandall/American-Sign-language, MIT), trained on the
// Kaggle Sign Language MNIST dataset, run with TensorFlow Lite Micro.
//
// Board/pins/options: config.h.   Serial commands (115200 baud, newline):
//   SNAP n     send n preprocessed 28x28 crops as "IMG <1568 hex>" lines (data collection)
//   PRED       run one burst and print the vote
//   LIVE 1|0   print the top class continuously (framing / lighting check)
//   INFO       print model + config
//   SAY word   test translate + speech
//   MODE sign|speech
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "config.h"
#include "vision.h"
#include "ui.h"
#include "name_snap.h"
#include "speech_out.h"
#include "speech_mode.h"

enum Mode { MODE_SIGN, MODE_SPEECH };
enum SignState { SPELLING, CHOOSING };

static Mode g_mode = MODE_SIGN;
static SignState g_state = SPELLING;
static char g_word[MAX_WORD_LEN + 1] = "";
static int g_suggest = -1;          // index into kNames while CHOOSING
static bool g_live = false;
static bool g_modelOk = false;

// ---------------------------------------------------------------- helpers
static void light(bool on) {
#if LIGHT_PIN >= 0
  ledcWrite(2, on ? LIGHT_DUTY : 0);
#endif
}

static void draw_sign_screen(const char* status) {
  lcd_row(0, "%s", status);
  lcd_row(1, "%s_", g_word);
  lcd_row(2, "");
  lcd_row(3, "%s", kHint);
}

static void show_spoken(const char* text) { lcd_row(2, "> %s", text); }

static void enter_mode(Mode m) {
  if (m == MODE_SIGN) {
    speech_mode_exit();
    if (!camera_start()) lcd_row(0, "CAMERA FAILED");
    light(true);
    g_state = SPELLING;
    draw_sign_screen(g_modelOk ? "SIGN MODE  ready" : "MODEL FAILED");
  } else {
    light(false);
    camera_stop();                 // free I2S0, LEDC0 and the frame buffers
    speech_mode_enter();
  }
  g_mode = m;
  Serial.printf("MODE %s\n", m == MODE_SIGN ? "sign" : "speech");
}

// ---------------------------------------------------------------- sign mode
static void commit_letter() {
  if (!g_modelOk || !camera_start()) return;
  lcd_row(0, "reading...");
  Burst b = capture_burst(BURST_FRAMES);
  if (b.label < 0) { lcd_row(0, "camera error"); return; }
  const char* L = kLabels[b.label];
  Serial.printf("PRED %s %.2f %d/%d second %s %.2f\n", L, b.conf, b.votes, b.frames,
                b.second >= 0 ? kLabels[b.second] : "-", b.secondConf);
  size_t n = strlen(g_word);
  if (n < MAX_WORD_LEN) { g_word[n] = L[0]; g_word[n + 1] = 0; }
  lcd_row(0, "[%s] %2d%% %d/%d  %s?%d%%", L, (int)(b.conf * 100), b.votes, b.frames,
          b.second >= 0 ? kLabels[b.second] : "-", (int)(b.secondConf * 100));
  lcd_row(1, "%s_", g_word);
}

static void backspace() {
  size_t n = strlen(g_word);
  if (n) g_word[n - 1] = 0;
  lcd_row(0, "deleted");
  lcd_row(1, "%s_", g_word);
}

static void speak_final(const char* w) {
  lcd_row(0, "Saying: %s", w);
  lcd_row(1, "%s", w);
  lcd_row(3, "");
  camera_stop();                    // RAM for WiFi + TLS on classic ESP32
  say_word(w, show_spoken);
  camera_start();
  g_word[0] = 0;
  g_state = SPELLING;
  lcd_row(0, "SIGN MODE  next word");
  lcd_row(1, "_");
  lcd_row(3, "%s", kHint);
}

static void finish_word() {
  if (!g_word[0]) return;
  g_suggest = snap_to_name(g_word);
  if (g_suggest < 0) { speak_final(g_word); return; }
  g_state = CHOOSING;
  lcd_row(0, "Raw : %s", g_word);
  lcd_row(1, "Name: %s", kNames[g_suggest]);
  lcd_row(2, "which one?");
  lcd_row(3, "%s", kChooseHint);
  Serial.printf("SUGGEST %s -> %s\n", g_word, kNames[g_suggest]);
}

static void sign_event(BtnEvent e) {
  if (g_state == CHOOSING) {
    if (e == EV_CAPTURE || e == EV_DONE) speak_final(kNames[g_suggest]);
    else if (e == EV_BACK) { char w[MAX_WORD_LEN + 1]; strcpy(w, g_word); speak_final(w); }
    return;
  }
  if (e == EV_CAPTURE) commit_letter();
  else if (e == EV_BACK) backspace();
  else if (e == EV_DONE) finish_word();
}

// ---------------------------------------------------------------- serial
static void cmd_snap(int n) {
  if (!camera_start()) { Serial.println("ERR camera"); return; }
  static const char* hex = "0123456789abcdef";
  uint8_t img[IMG_PIXELS];
  char line[4 + IMG_PIXELS * 2 + 1];
  memcpy(line, "IMG ", 4);
  for (int k = 0; k < n; k++) {
    uint32_t t = millis();
    if (!grab_crop(img)) { Serial.println("ERR grab"); continue; }
    for (int i = 0; i < IMG_PIXELS; i++) {
      line[4 + 2 * i] = hex[img[i] >> 4];
      line[5 + 2 * i] = hex[img[i] & 15];
    }
    line[sizeof line - 1] = 0;
    Serial.println(line);
    while (millis() - t < SNAP_INTERVAL_MS) delay(1);
  }
  Serial.println("END");
}

static void handle_serial() {
  static String buf;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') { if (buf.length() < 80) buf += c; continue; }
    String cmd = buf; buf = "";
    cmd.trim();
    String up = cmd; up.toUpperCase();
    if (up.startsWith("SNAP")) cmd_snap(max(1, (int)up.substring(4).toInt()));
    else if (up == "PRED") {
      if (!g_modelOk || !camera_start()) { Serial.println("ERR model/camera"); continue; }
      Burst b = capture_burst(BURST_FRAMES);
      if (b.label < 0) { Serial.println("ERR grab"); continue; }
      Serial.printf("PRED %s %.2f %d/%d second %s %.2f\n", kLabels[b.label], b.conf, b.votes,
                    b.frames, b.second >= 0 ? kLabels[b.second] : "-", b.secondConf);
    } else if (up.startsWith("LIVE")) g_live = up.endsWith("1");
    else if (up == "INFO") {
      Serial.printf("INFO labels=");
      for (int i = 0; i < kNumLabels; i++) Serial.print(kLabels[i]);
      Serial.printf(" model=%u crop=%d off=%d,%d mirror=%d stretch=%d psram=%d heap=%u\n",
                    g_model_len, CROP_SIZE, CROP_X_OFFSET, CROP_Y_OFFSET, MIRROR_X,
                    CONTRAST_STRETCH, psramFound(), ESP.getFreeHeap());
    } else if (up.startsWith("SAY ")) {
      String w = cmd.substring(4); w.toUpperCase();
      speak_final(w.c_str());
    } else if (up == "MODE SPEECH") enter_mode(MODE_SPEECH);
    else if (up == "MODE SIGN") enter_mode(MODE_SIGN);
    else if (cmd.length()) Serial.println("ERR unknown command");
  }
}

// ---------------------------------------------------------------- main
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // camera inrush trips brown-out on USB power
  Serial.begin(115200);
  delay(300);
  Serial.println("\nASL speller booting");
  if (!psramFound()) Serial.println("WARN no PSRAM: frame buffers in DRAM, TTS disabled");
#if LIGHT_PIN >= 0
  ledcSetup(2, 5000, 8);           // channel 2 / timer 1 (camera uses channel 0 / timer 0)
  ledcAttachPin(LIGHT_PIN, 2);
#endif
  buttons_begin();
  lcd_begin();
  lcd_row(0, "ASL speller");
  lcd_row(1, "loading model...");
  g_modelOk = classifier_init();
  enter_mode(MODE_SIGN);
}

void loop() {
  handle_serial();
  BtnEvent e = buttons_poll();
  if (e == EV_MODE) enter_mode(g_mode == MODE_SIGN ? MODE_SPEECH : MODE_SIGN);
  else if (g_mode == MODE_SIGN) {
    if (e != EV_NONE) sign_event(e);
    if (g_live && g_modelOk) {
      uint8_t img[IMG_PIXELS];
      float p[MAX_LABELS];
      if (grab_crop(img)) {
        int k = classify(img, p);
        if (k >= 0) Serial.printf("LIVE %s %.2f\n", kLabels[k], p[k]);
      }
    }
  } else {
    speech_mode_loop();
  }
  delay(5);
}
