#pragma once
// ============================================================================
//  Everything you are likely to change lives in this file.
// ============================================================================

// ---- 1. Board -----------------------------------------------------------
// Pick ONE, or leave all commented to auto-pick (S3 chip -> ESP32S3_EYE pinout,
// classic ESP32 -> AI_THINKER). Check the chip first: the Arduino IDE prints
// "Chip is ESP32-S3" / "Chip is ESP32-D0WD" when uploading.
// #define CAMERA_MODEL_AI_THINKER      // classic ESP32-CAM (most common)
// #define CAMERA_MODEL_ESP32S3_EYE     // ESP32-S3-EYE, Freenove ESP32-S3-WROOM CAM
// #define CAMERA_MODEL_XIAO_ESP32S3    // Seeed XIAO ESP32S3 Sense

#if !defined(CAMERA_MODEL_AI_THINKER) && !defined(CAMERA_MODEL_ESP32S3_EYE) && \
    !defined(CAMERA_MODEL_XIAO_ESP32S3)
#  if defined(CONFIG_IDF_TARGET_ESP32S3)
#    define CAMERA_MODEL_ESP32S3_EYE
#  else
#    define CAMERA_MODEL_AI_THINKER
#  endif
#endif

// ---- 2. Peripheral pins (-1 = not connected) -----------------------------
#if defined(CAMERA_MODEL_AI_THINKER)
// Classic ESP32-CAM has only GPIO 2, 4, 12, 13, 14, 15 free (16 = PSRAM,
// 0 = camera clock, 1/3 = USB serial). So: ONE button (gesture mode) and
// the camera's I2S0 is taken, so the amp uses I2S1.
#  define LCD_SDA        14
#  define LCD_SCL        15
#  define BUTTON_SINGLE  1      // one button: tap / double-tap / hold
#  define BTN_CAPTURE    13
#  define BTN_BACK       -1
#  define BTN_DONE       -1
#  define BTN_MODE       -1
#  define I2S_BCLK       12     // shared with INMP441 SCK
#  define I2S_LRC         2     // shared with INMP441 WS
#  define AMP_DOUT        4     // onboard flash LED will flicker while speaking
#  define MIC_DIN        -1     // no free pin; GPIO3 only if you give up serial RX
#  define LIGHT_PIN      -1     // wire the frame LEDs straight to 3V3/5V instead
#elif defined(CAMERA_MODEL_XIAO_ESP32S3)
#  define LCD_SDA         5     // D4
#  define LCD_SCL         6     // D5
#  define BUTTON_SINGLE   0
#  define BTN_CAPTURE     1     // D0
#  define BTN_BACK        2     // D1
#  define BTN_DONE        3     // D2
#  define BTN_MODE        4     // D3
#  define I2S_BCLK        7     // D8
#  define I2S_LRC         8     // D9
#  define AMP_DOUT        9     // D10
#  define MIC_DIN        44     // D7
#  define LIGHT_PIN      43     // D6
#else  // ESP32S3_EYE / Freenove S3 CAM pinout
#  define LCD_SDA         1
#  define LCD_SCL         2
#  define BUTTON_SINGLE   0
#  define BTN_CAPTURE    14
#  define BTN_BACK       21
#  define BTN_DONE       47
#  define BTN_MODE       38
#  define I2S_BCLK       39
#  define I2S_LRC        40
#  define AMP_DOUT       41
#  define MIC_DIN        42
#  define LIGHT_PIN       3
#endif
#define AMP_I2S_PORT  I2S_NUM_1   // camera owns I2S0 on classic ESP32
#define LIGHT_DUTY    160         // 0-255 PWM if LIGHT_PIN is used

// ---- 3. Image preprocessing ---------------------------------------------
// !! Do NOT change these after collecting training data: the model learns
// !! exactly what this pipeline produces.
#define CROP_SIZE        120   // square side in QQVGA pixels (<=120)
#define CROP_X_OFFSET      0   // shift the square right (+) / left (-)
#define CROP_Y_OFFSET      0   // shift the square down (+) / up (-)
#define MIRROR_X           0   // 1 = flip left/right before the model
#define CONTRAST_STRETCH   1   // 1 = stretch 2nd..98th percentile to 0..255
#define CAM_VFLIP          0

// ---- 4. Letter commit ----------------------------------------------------
#define BURST_FRAMES       5
#define BURST_GAP_MS      60
#define MAX_WORD_LEN      16
#define SNAP_MIN_LEN       3   // only snap words this long or longer to names
#define SNAP_INTERVAL_MS 120   // spacing between frames in "SNAP n" (data collection)

// ---- 5. Translate + speech ----------------------------------------------
#define USE_NETWORK        1       // 0 = LCD only, no WiFi/translate/TTS
#define TARGET_LANG      "en"      // Google Translate target; "en" skips translation
#define TTS_LANG_CODE    "en-US"   // Cloud TTS voice language, e.g. fr-FR, es-ES, hi-IN
#define TTS_SAMPLE_RATE  16000
#define TTS_VOLUME       0.6f
// WiFi + API key live in secrets.h (copy secrets.example.h; it is gitignored).
