#pragma once
// Camera -> centre square -> grayscale 28x28 -> int8 CNN.
// Included once, from asl_sign.ino.
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <esp_heap_caps.h>
#include "esp_camera.h"
#include "config.h"
#include "camera_pins.h"
#include "model_data.h"

static const int IMG = 28;
static const int IMG_PIXELS = IMG * IMG;
static const int MAX_LABELS = 32;

// ---------------------------------------------------------------- camera
// The frame light follows the camera. LEDC channel 2 = timer 1; the camera's
// XCLK uses channel 0 / timer 0.
static bool g_cameraOn = false;
static const int kLightChannel = 2;

static void light(bool on) {
#if LIGHT_PIN >= 0
  static bool attached = false;
  if (!attached) { ledcSetup(kLightChannel, 5000, 8); ledcAttachPin(LIGHT_PIN, kLightChannel); attached = true; }
  ledcWrite(kLightChannel, on ? LIGHT_DUTY : 0);
#endif
}

static bool camera_start() {
  if (g_cameraOn) return true;
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_GRAYSCALE;
  c.frame_size = FRAMESIZE_QQVGA;          // 160x120 is plenty for 28x28
  c.fb_count = 2;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("ERR camera init 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  s->set_exposure_ctrl(s, 1);   // auto exposure on
  s->set_gain_ctrl(s, 1);
  s->set_whitebal(s, 1);
  s->set_vflip(s, CAM_VFLIP);
  g_cameraOn = true;
  light(true);
  // let auto-exposure settle
  for (int i = 0; i < 5; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  return true;
}

static void camera_stop() {
  if (!g_cameraOn) return;
  esp_camera_deinit();          // frees frame buffers and I2S0 / LEDC0
  g_cameraOn = false;
  light(false);
}

// ---------------------------------------------------------------- preprocess
// Same function produces training crops (SNAP) and inference input, so the
// model always sees exactly what it was trained on.
static void preprocess(const uint8_t* src, int w, int h, uint8_t* out) {
  int side = CROP_SIZE;
  if (side > w) side = w;
  if (side > h) side = h;
  int x0 = (w - side) / 2 + CROP_X_OFFSET;
  int y0 = (h - side) / 2 + CROP_Y_OFFSET;
  x0 = constrain(x0, 0, w - side);
  y0 = constrain(y0, 0, h - side);

  // area-average downsample (box filter), no aliasing
  for (int dy = 0; dy < IMG; dy++) {
    int sy0 = y0 + dy * side / IMG, sy1 = y0 + (dy + 1) * side / IMG;
    for (int dx = 0; dx < IMG; dx++) {
      int sx0 = x0 + dx * side / IMG, sx1 = x0 + (dx + 1) * side / IMG;
      uint32_t sum = 0, n = 0;
      for (int y = sy0; y < sy1; y++) {
        const uint8_t* row = src + y * w;
        for (int x = sx0; x < sx1; x++) { sum += row[x]; n++; }
      }
      int ox = MIRROR_X ? (IMG - 1 - dx) : dx;
      out[dy * IMG + ox] = n ? sum / n : 0;
    }
  }

#if CONTRAST_STRETCH
  uint16_t hist[256] = {0};
  for (int i = 0; i < IMG_PIXELS; i++) hist[out[i]]++;
  int cut = IMG_PIXELS * 2 / 100, acc = 0, lo = 0, hi = 255;
  for (int v = 0; v < 256; v++) { acc += hist[v]; if (acc > cut) { lo = v; break; } }
  acc = 0;
  for (int v = 255; v >= 0; v--) { acc += hist[v]; if (acc > cut) { hi = v; break; } }
  if (hi - lo >= 8) {
    for (int i = 0; i < IMG_PIXELS; i++) {
      int v = (out[i] - lo) * 255 / (hi - lo);
      out[i] = constrain(v, 0, 255);
    }
  }
#endif
}

// Grab one fresh frame and preprocess it. Returns false on camera error.
static bool grab_crop(uint8_t* out) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  bool ok = fb->format == PIXFORMAT_GRAYSCALE;
  if (ok) preprocess(fb->buf, fb->width, fb->height, out);
  esp_camera_fb_return(fb);
  return ok;
}

// ---------------------------------------------------------------- model
static tflite::MicroInterpreter* g_interp = nullptr;
static TfLiteTensor* g_in = nullptr;
static TfLiteTensor* g_out = nullptr;
static int8_t g_qlut[256];          // pixel 0..255 -> model int8 input
static const int kArenaSize = 96 * 1024;

static bool classifier_init() {
  static tflite::MicroErrorReporter reporter;
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR model schema version");
    return false;
  }
  uint8_t* arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM);  // slower but works
  if (!arena) { Serial.println("ERR no memory for tensor arena"); return false; }

  // Exactly the ops the stock and fine-tuned models use (train_asl.py FIRMWARE_OPS).
  static tflite::MicroMutableOpResolver<5> ops;
  ops.AddConv2D();
  ops.AddMaxPool2D();
  ops.AddReshape();
  ops.AddFullyConnected();
  ops.AddSoftmax();

  static tflite::MicroInterpreter interp(model, ops, arena, kArenaSize, &reporter);
  if (interp.AllocateTensors() != kTfLiteOk) { Serial.println("ERR AllocateTensors"); return false; }
  g_interp = &interp;
  g_in = interp.input(0);
  g_out = interp.output(0);
  if (g_in->type != kTfLiteInt8 || g_in->bytes != IMG_PIXELS) {
    Serial.println("ERR model input must be int8 28x28x1");
    return false;
  }
  if (g_out->dims->data[g_out->dims->size - 1] != kNumLabels || kNumLabels < 2 || kNumLabels > MAX_LABELS) {
    Serial.println("ERR label count does not match model output");
    return false;
  }
  // Training normalisation is x/127.5 - 1  (same as the upstream notebook).
  float scale = g_in->params.scale;
  int zp = g_in->params.zero_point;
  for (int p = 0; p < 256; p++) {
    int q = lroundf((p / 127.5f - 1.0f) / scale) + zp;
    g_qlut[p] = constrain(q, -128, 127);
  }
  Serial.printf("model ok: %u bytes, %d labels, arena used %u / %d\n",
                g_model_len, kNumLabels, (unsigned)interp.arena_used_bytes(), kArenaSize);
  return true;
}

// Runs the CNN on one 28x28 crop. Fills probs[kNumLabels]; returns argmax.
static int classify(const uint8_t* img, float* probs) {
  for (int i = 0; i < IMG_PIXELS; i++) g_in->data.int8[i] = g_qlut[img[i]];
  if (g_interp->Invoke() != kTfLiteOk) return -1;
  float s = g_out->params.scale;
  int zp = g_out->params.zero_point;
  int best = 0;
  for (int i = 0; i < kNumLabels; i++) {
    probs[i] = (g_out->data.int8[i] - zp) * s;
    if (probs[i] > probs[best]) best = i;
  }
  return best;
}

// ---------------------------------------------------------------- burst vote
struct Burst {
  int label = -1;      // winning class (majority of per-frame argmax); -1 = no frames
  float conf = 0;      // mean probability of the winner over the burst
  int votes = 0, frames = 0;
  int second = 0;      // runner-up by mean probability
  float secondConf = 0;
};

static Burst capture_burst(int n) {
  Burst b;
  int votes[MAX_LABELS] = {0};
  float sum[MAX_LABELS] = {0};
  float probs[MAX_LABELS];
  uint8_t img[IMG_PIXELS];
  for (int f = 0; f < n; f++) {
    if (f) delay(BURST_GAP_MS);
    if (!grab_crop(img)) continue;
    int k = classify(img, probs);
    if (k < 0) continue;
    votes[k]++;
    for (int i = 0; i < kNumLabels; i++) sum[i] += probs[i];
    b.frames++;
  }
  if (!b.frames) return b;
  b.label = 0;
  for (int i = 1; i < kNumLabels; i++)
    if (votes[i] > votes[b.label] || (votes[i] == votes[b.label] && sum[i] > sum[b.label])) b.label = i;
  b.second = b.label == 0 ? 1 : 0;
  for (int i = 0; i < kNumLabels; i++)
    if (i != b.label && sum[i] > sum[b.second]) b.second = i;
  b.votes = votes[b.label];
  b.conf = sum[b.label] / b.frames;
  b.secondConf = sum[b.second] / b.frames;
  return b;
}
