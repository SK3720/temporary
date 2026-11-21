#pragma once
// Speech mode (INMP441 mic -> speech-to-text -> translate -> LCD + speaker).
//
// PASTE YOUR TEAM'S EXISTING SPEECH PIPELINE HERE. The camera is fully shut
// down (esp_camera_deinit) before speech_mode_enter() runs, so on a classic
// ESP32 you get I2S0 back and all the PSRAM. Mic and amp can share BCLK/WS
// (I2S_BCLK / I2S_LRC in config.h) with the mic on MIC_DIN, full duplex.
#include "ui.h"

static void speech_mode_enter() {
  lcd_row(0, "SPEECH MODE");
  lcd_row(1, "(pipeline goes in");
  lcd_row(2, " speech_mode.h)");
  lcd_row(3, "%s", kModeHint);
}

static void speech_mode_loop() {}

static void speech_mode_exit() {
  // stop the mic I2S driver here (i2s_driver_uninstall) so sign mode can
  // reclaim I2S0 for the camera on a classic ESP32.
}
