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

static Mode g_mode = MODE_SIGN;
static char g_word[MAX_WORD_LEN + 1] = "";
static int g_suggest = -1;          // >= 0 while choosing between g_word and kNames[g_suggest]
static char g_status[48] = "";
static bool g_live = false;
static bool g_modelOk = false;

// ---------------------------------------------------------------- screen
// Rows 0, 1 and 3 are drawn from state; row 2 holds the last spoken word.
static void draw() {
  if (g_suggest >= 0) {
    lcd_row(0, "Raw : %s", g_word);
    lcd_row(1, "Name: %s", kNames[g_suggest]);
    lcd_row(2, "which one?");
    lcd_row(3, "%s", kChooseHint);
  } else {
    lcd_row(0, "%s", g_status);
    lcd_row(1, "%s_", g_word);
    lcd_row(3, "%s", kHint);
  }
}

static void set_status(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_status, sizeof g_status, fmt, ap);
  va_end(ap);
  draw();
}

static void enter_mode(Mode m) {
  static bool started = false;
  if (started && m == g_mode) return;
  if (started && g_mode == MODE_SPEECH) speech_mode_exit();
  started = true;
  g_mode = m;
  if (m == MODE_SIGN) {
    g_suggest = -1;
    lcd_row(2, "");
    set_status(!g_modelOk ? "MODEL FAILED" : camera_start() ? "SIGN MODE  ready" : "CAMERA FAILED");
  } else {
    camera_stop();                 // free I2S0, LEDC0 and the frame buffers
    speech_mode_enter();
  }
  Serial.printf("MODE %s\n", m == MODE_SIGN ? "sign" : "speech");
}

// ---------------------------------------------------------------- sign mode
// One burst + majority vote. The PRED line format is parsed by collect.py quiz.
static bool read_letter(Burst& b) {
  if (!g_modelOk || !camera_start()) { Serial.println("ERR model/camera"); return false; }
  b = capture_burst(BURST_FRAMES);
  if (b.label < 0) { Serial.println("ERR grab"); return false; }
  Serial.printf("PRED %s %.2f %d/%d second %s %.2f\n", kLabels[b.label], b.conf, b.votes,
                b.frames, kLabels[b.second], b.secondConf);
  return true;
}

static void commit_letter() {
  set_status("reading...");
  Burst b;
  if (!read_letter(b)) { set_status("camera error"); return; }
  size_t n = strlen(g_word);
  if (n < MAX_WORD_LEN) { g_word[n] = kLabels[b.label][0]; g_word[n + 1] = 0; }
  set_status("[%s] %2d%% %d/%d  %s?%d%%", kLabels[b.label], (int)(b.conf * 100), b.votes,
             b.frames, kLabels[b.second], (int)(b.secondConf * 100));
}

static void speak_final(const char* w) {
  g_suggest = -1;
  lcd_row(0, "Saying: %s", w);
  lcd_row(1, "%s", w);
  lcd_row(3, "");
  say_word(w);                      // shows the spoken text on row 2
  g_word[0] = 0;
  set_status("SIGN MODE  next word");
}

static void sign_event(BtnEvent e) {
  if (g_suggest >= 0) {             // choosing: SIGN/DONE = suggested name, DEL = raw spelling
    if (e == EV_BACK) speak_final(g_word);
    else if (e == EV_CAPTURE || e == EV_DONE) speak_final(kNames[g_suggest]);
    return;
  }
  if (e == EV_CAPTURE) {
    commit_letter();
  } else if (e == EV_BACK) {
    size_t n = strlen(g_word);
    if (n) g_word[n - 1] = 0;
    set_status("deleted");
  } else if (e == EV_DONE && g_word[0]) {
    g_suggest = snap_to_name(g_word);
    if (g_suggest < 0) { speak_final(g_word); return; }
    Serial.printf("SUGGEST %s -> %s\n", g_word, kNames[g_suggest]);
    draw();
  }
}

static void live_tick() {
  uint8_t img[IMG_PIXELS];
  float p[MAX_LABELS];
  if (!g_modelOk || !grab_crop(img)) return;
  int k = classify(img, p);
  if (k >= 0) Serial.printf("LIVE %s %.2f\n", kLabels[k], p[k]);
}

// ---------------------------------------------------------------- serial
static void cmd_snap(int n) {
  if (!camera_start()) { Serial.println("ERR camera"); return; }
  static const char* hex = "0123456789abcdef";
  uint8_t img[IMG_PIXELS];
  char row[IMG * 2];
  for (int k = 0; k < n; k++) {
    uint32_t t = millis();
    if (!grab_crop(img)) { Serial.println("ERR grab"); continue; }
    Serial.print("IMG ");
    for (int y = 0; y < IMG; y++) {        // one row at a time keeps the stack small
      for (int x = 0; x < IMG; x++) {
        uint8_t v = img[y * IMG + x];
        row[2 * x] = hex[v >> 4];
        row[2 * x + 1] = hex[v & 15];
      }
      Serial.write((const uint8_t*)row, sizeof row);
    }
    Serial.println();
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
    String up = buf;
    buf = "";
    up.trim();
    up.toUpperCase();
    Burst b;
    if (up.startsWith("SNAP")) cmd_snap(max(1, (int)up.substring(4).toInt()));
    else if (up == "PRED") read_letter(b);
    else if (up.startsWith("LIVE")) g_live = up.endsWith("1");
    else if (up == "INFO") {
      Serial.print("INFO labels=");
      for (int i = 0; i < kNumLabels; i++) Serial.print(kLabels[i]);
      Serial.printf(" model=%u crop=%d off=%d,%d mirror=%d stretch=%d psram=%d heap=%u\n",
                    g_model_len, CROP_SIZE, CROP_X_OFFSET, CROP_Y_OFFSET, MIRROR_X,
                    CONTRAST_STRETCH, psramFound(), ESP.getFreeHeap());
    } else if (up.startsWith("SAY ")) speak_final(up.substring(4).c_str());
    else if (up == "MODE SPEECH") enter_mode(MODE_SPEECH);
    else if (up == "MODE SIGN") enter_mode(MODE_SIGN);
    else if (up.length()) Serial.println("ERR unknown command");
  }
}

// ---------------------------------------------------------------- main
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // camera inrush trips brown-out on USB power
  Serial.begin(115200);
  delay(300);
  Serial.println("\nASL speller booting");
  if (!psramFound()) Serial.println("WARN no PSRAM: frame buffers in DRAM, TTS disabled");
  buttons_begin();
  lcd_begin();
  lcd_row(0, "ASL speller");
  lcd_row(1, "loading model...");
  g_modelOk = classifier_init();
  network_begin();                 // connects in the background
  enter_mode(MODE_SIGN);
}

void loop() {
  handle_serial();
  BtnEvent e = buttons_poll();
  if (e == EV_MODE) enter_mode(g_mode == MODE_SIGN ? MODE_SPEECH : MODE_SIGN);
  else if (g_mode == MODE_SPEECH) speech_mode_loop();
  else {
    if (e != EV_NONE) sign_event(e);
    if (g_live) live_tick();
  }
  delay(5);
}
