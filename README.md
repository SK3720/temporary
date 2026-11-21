# ASL fingerspelling → spoken word, on one ESP32

The ESP32 does all the sign recognition. No frames leave the board.

```
camera (QQVGA gray) → centre square → 28×28 area-average → contrast stretch
  → x/127.5−1 → int8 CNN (TFLite Micro) → 5-frame majority vote → letter
  → word (buttons) → name snap (optional) → Google Translate → LCD + speech
```

The model and approach come from the organizers' recommended repo [1]. It's a Keras CNN trained on Sign Language MNIST [2] and run with TensorFlow Lite for Microcontrollers [3]. We fine-tune it on our own camera's crops and add J and Z as static classes.

## Layout

| Path | What |
|---|---|
| `firmware/asl_sign/` | Arduino sketch. **Start in `config.h`.** |
| `firmware/asl_sign/model_data.cpp` | Generated. It ships with the **stock 24-class model** from [1], so you can flash it right away. |
| `firmware/asl_sign/speech_mode.h` | Paste the team's existing speech pipeline here. |
| `firmware/asl_sign/speech_out.h` | Translate + TTS for finished words. If you already have a path, call it from `say_word()`, which already stops the camera around network calls. |
| `tools/collect.py` | Serial tool: `preview`, `collect`, `sheet`, `quiz`. |
| `training/train_asl.py`, `training/colab.ipynb` | Fine-tune, quantize to int8, write `model_data.cpp`. |
| `tools/tflite_to_c.py` | `.tflite` → `model_data.cpp`, with the labels embedded. |
| `tools/make_names.py` | Rebuilds `names.h`: about 1,000 common first names. |

## Setup (Arduino IDE)

1. Boards Manager: **esp32 by Espressif, version 2.0.17**. Use a 2.0.x version: the TFLite library below doesn't build on 3.x.
2. Library Manager: **TensorFlowLite_ESP32** (1.0.0) and **LiquidCrystal I2C** (Frank de Brabander).
3. Board settings:
   - **AI-Thinker ESP32-CAM**: board "AI Thinker ESP32-CAM".
   - **ESP32-S3**: board "ESP32S3 Dev Module", PSRAM "OPI PSRAM", partition "Huge APP".
4. Copy `secrets.example.h` to `secrets.h` (gitignored) and add the WiFi details and Google API key. Set `USE_NETWORK 0` in `config.h` to test without them.
5. Open Serial Monitor at 115200. You should see `model ok: … arena used …` and `MODE sign`.

All three board variants compile on esp32 core 2.0.17: AI-Thinker, ESP32-S3-EYE/Freenove, and XIAO S3.

## Which chip do you have? (decides the wiring)

When you upload, the IDE prints `Chip is ESP32-D0WD…` (classic) or `ESP32-S3`. The defaults in `config.h` switch automatically.

| | Classic ESP32-CAM (AI-Thinker) | ESP32-S3 (EYE / Freenove pinout) |
|---|---|---|
| Free GPIO | only 2, 4, 12, 13, 14, 15 | plenty |
| LCD SDA/SCL | 14 / 15 | 1 / 2 |
| Buttons | **one button on 13**: tap = add letter, double-tap = delete, hold 0.8 s = done, hold 2.5 s = switch mode | CAPTURE 14, DEL 21, DONE 47, MODE 38 (to GND) |
| I2S BCLK / LRC | 12 / 2 (shared by amp and mic) | 39 / 40 |
| Amp DIN (MAX98357A) | 4 (the flash LED flickers while speaking) | 41 |
| Mic SD (INMP441) | **no free pin.** GPIO3 works only if you give up serial RX, so use it in the final build only. | 42 |
| I2S | The camera takes I2S0, so the amp uses I2S1. | Separate camera peripheral |

- The frame LEDs go straight to 3V3/5V through a resistor, so they don't need a pin.
- Sign mode and speech mode run one at a time. Leaving sign mode calls `esp_camera_deinit()`.
- Speaking a word also shuts the camera down, because WiFi + TLS + camera doesn't fit in RAM on a classic ESP32.
- PSRAM is required for the frame buffers and the TTS audio. AI-Thinker and the S3 boards above all have it. Check that `INFO` prints `psram=1`.

## Run sheet

### 1. Confirm inference with the stock model (13:45–14:15)

Flash as-is, then run:

```bash
pip install pyserial
```
```bash
python tools/collect.py preview
```

This draws the live 28×28 crop the model sees in your terminal. Type `PRED` or `LIVE 1` in Serial Monitor to see predictions.

The stock model has no J/Z and was trained on different-looking images, so expect weak accuracy here. This step only proves the pipeline works.

### 2. Frame + data (14:15–14:45)

Build the cardboard box: plain backdrop, hand about 25–30 cm from the lens, LEDs lighting the hand from the front.

- **Before you collect any data**, use `preview` to adjust `CROP_SIZE` / `CROP_X_OFFSET` / `CROP_Y_OFFSET` / `MIRROR_X` so the hand fills the square. **Then freeze them.** The model learns exactly what this preprocessing produces.
- Collect data for each teammate:

```bash
python tools/collect.py collect --signer alice
```

That's 26 letters × 30 crops, about 4 minutes per person. Move your hand slightly while it records. For J and Z, hold the shape from the middle of the motion. Re-run with `--letters JZ` for top-ups.

- Check the data by opening `data/sheet.pgm` in Preview after running:

```bash
python tools/collect.py sheet
```

### 3. Fine-tune (14:45–15:15)

Open `training/colab.ipynb` in Colab with a GPU runtime. Upload `training/train_asl.py`, `tools/tflite_to_c.py` and `data/crops.csv`.

- **Cell 2, `--holdout <one teammate>`**: prints per-letter accuracy and the top confusions on a signer the model never saw. **This is your judge number.**
- **Cell 3**: trains on everyone and downloads `model_data.cpp`. Drop it into `firmware/asl_sign/` and re-flash.

The labels travel inside that file, so J/Z appear on the device automatically.

If `kagglehub` can't fetch Sign MNIST, download `sign_mnist_train.csv` / `sign_mnist_test.csv` from Kaggle, upload them, and add `--mnist-dir .`. If you have no MNIST data at all, `--no-mnist` trains on your crops only.

Each run takes a few minutes on a Colab GPU. The script was tested end to end locally with TF 2.15.

### 4. UX + speech (15:15–16:00)

The flow is already built in:

- **Capture** runs a 5-frame burst and a majority vote. The LCD shows `[A] 87% 4/5  E?6%`: letter, confidence, votes, runner-up.
- **Delete** removes the last letter.
- **Done** snaps the word to a common name if one is within one edit. Letter swaps the model often confuses (AEMNST, UVR, GH, IJ) count as closer matches. The LCD shows `Raw : JOHM` and `Name: JOHN` and the user picks one. The word is then translated (`TARGET_LANG`, where `"en"` skips translation), shown on the LCD and spoken.

TTS uses Google Cloud Text-to-Speech, which returns LINEAR16 audio. That goes straight to the MAX98357A, so no MP3 decoder is needed. Test it from Serial Monitor with `SAY JOHN`.

### 5. Stranger test (16:00)

```bash
python tools/collect.py quiz --n 26
```

It prompts random letters, runs `PRED` for each, and prints accuracy and misses.

## Serial commands (115200, newline)

| Command | Does |
|---|---|
| `SNAP n` | Sends n preprocessed crops as `IMG <1568 hex chars>`, then `END` |
| `PRED` | Runs one burst and prints `PRED A 0.87 4/5 second E 0.06` |
| `LIVE 1` / `LIVE 0` | Streams top-1 continuously |
| `INFO` | Labels, crop settings, PSRAM, free heap |
| `SAY word` | Translate + speak test |
| `MODE sign` / `MODE speech` | Switches mode |

## Judge Q&A

- **Why not detect letters continuously?** Double letters (Anna, Matt) and transition frames. A button-triggered 5-frame burst with a majority vote fixes both.
- **How did you handle J and Z?** They're motion letters, so Sign MNIST has no classes for them. We added static mid-motion classes from our own data.
- **How does it do on unseen hands?** Give the held-out signer number from Colab cell 2, plus the `quiz` result.
- **Why on-device?** No network round-trip and it works offline. It also meets the "microcontroller as core" and "no outside hardware" rules. The network is used only to translate and speak the finished word.
- **Why did accuracy improve over the stock model?** Three reasons:
  - A fixed frame and lighting make our images look like the dataset.
  - We fine-tuned on crops that went through the exact on-device preprocessing.
  - A left/right flip augmentation handles left-handed signers.

## References (IEEE)

[1] J. Randall, "American-Sign-language," GitHub repository, 2023. [Online]. Available: https://github.com/jonathanrandall/American-Sign-language

[2] DataMunge, "Sign Language MNIST," Kaggle, 2017. [Online]. Available: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

[3] R. David *et al.*, "TensorFlow Lite Micro: Embedded machine learning for TinyML systems," in *Proc. Machine Learning and Systems (MLSys)*, 2021.

The stock model bytes in `model_data.cpp` and `models/stock_asl24.tflite` come from [1], MIT License, © 2023 Jonathan. The name list comes from US Social Security baby-name data (hadley/data-baby-names) plus hand-added international names.
