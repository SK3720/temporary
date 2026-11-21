#pragma once
// Buttons + 20x4 I2C character LCD. Included once, from asl_sign.ino.
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "config.h"

// ---------------------------------------------------------------- buttons
enum BtnEvent { EV_NONE, EV_CAPTURE, EV_BACK, EV_DONE, EV_MODE };

#if BUTTON_SINGLE
// One button: tap = capture, double-tap = backspace, hold 0.8 s = done,
// hold 2.5 s = switch mode.
static const uint32_t DOUBLE_MS = 350, HOLD_MS = 800, MODE_MS = 2500;
static bool s_down = false, s_pendingTap = false;
static uint32_t s_downAt = 0, s_upAt = 0, s_lastEdge = 0;

static void buttons_begin() { pinMode(BTN_CAPTURE, INPUT_PULLUP); }

static BtnEvent buttons_poll() {
  uint32_t now = millis();
  bool down = digitalRead(BTN_CAPTURE) == LOW;
  if (down != s_down && now - s_lastEdge > 25) {
    s_lastEdge = now;
    s_down = down;
    if (down) {
      s_downAt = now;
    } else {
      uint32_t held = now - s_downAt;
      if (held >= MODE_MS) { s_pendingTap = false; return EV_MODE; }
      if (held >= HOLD_MS) { s_pendingTap = false; return EV_DONE; }
      if (s_pendingTap && now - s_upAt < DOUBLE_MS) { s_pendingTap = false; return EV_BACK; }
      s_pendingTap = true;
      s_upAt = now;
    }
  }
  if (s_pendingTap && !s_down && now - s_upAt >= DOUBLE_MS) {
    s_pendingTap = false;
    return EV_CAPTURE;
  }
  return EV_NONE;
}
static const char* kHint = "tap=add 2x=del hold=ok";
static const char* kChooseHint = "tap=name  2x=raw";
#else
static const int kPins[4] = {BTN_CAPTURE, BTN_BACK, BTN_DONE, BTN_MODE};
static const BtnEvent kEv[4] = {EV_CAPTURE, EV_BACK, EV_DONE, EV_MODE};
static bool s_state[4];
static uint32_t s_edge[4];

static void buttons_begin() {
  for (int i = 0; i < 4; i++) if (kPins[i] >= 0) pinMode(kPins[i], INPUT_PULLUP);
}

static BtnEvent buttons_poll() {
  uint32_t now = millis();
  for (int i = 0; i < 4; i++) {
    if (kPins[i] < 0) continue;
    bool down = digitalRead(kPins[i]) == LOW;
    if (down != s_state[i] && now - s_edge[i] > 30) {
      s_edge[i] = now;
      s_state[i] = down;
      if (down) return kEv[i];
    }
  }
  return EV_NONE;
}
static const char* kHint = "SIGN  DEL  DONE";
static const char* kChooseHint = "SIGN=name  DEL=raw";
#endif

// ---------------------------------------------------------------- LCD
static LiquidCrystal_I2C* g_lcd = nullptr;

static void lcd_begin() {
  Wire.begin(LCD_SDA, LCD_SCL);
  uint8_t addr = 0;
  for (uint8_t a : {0x27, 0x3F, 0x26, 0x20}) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { addr = a; break; }
  }
  if (!addr) { Serial.println("WARN no LCD found on I2C; continuing on serial only"); return; }
  g_lcd = new LiquidCrystal_I2C(addr, 20, 4);
  g_lcd->init();
  g_lcd->backlight();
  Serial.printf("LCD at 0x%02X\n", addr);
}

// Print one full row (padded/truncated to 20 chars). Non-ASCII -> '?'.
static void lcd_row(int row, const char* fmt, ...) {
  char buf[64];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (!g_lcd) return;
  char line[21];
  int i = 0;
  for (; i < 20 && buf[i]; i++) line[i] = (buf[i] >= 32 && buf[i] < 127) ? buf[i] : '?';
  for (; i < 20; i++) line[i] = ' ';
  line[20] = 0;
  g_lcd->setCursor(0, row);
  g_lcd->print(line);
}
