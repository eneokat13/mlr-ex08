# Leaf classifier — live camera classification in the browser

**Live page: <https://eneokat13.github.io/mlr-ex08/>**

Classifies a leaf as `apple`, `cherry`, `chestnut` or `maple` straight from the phone's
rear camera. Everything runs on the device: the video is never uploaded anywhere, and
there is no server, no build step and no framework — one `index.html` of plain
JavaScript plus the ONNX model, served as static files by GitHub Pages.

Scan the QR code at the bottom of the page with a phone camera to open it, or use the
text link next to it. The QR code is drawn by an encoder written into the page itself
(byte mode, error-correction level M, version 3), not fetched from any QR service.

## Run it

Open the [live page](https://eneokat13.github.io/mlr-ex08/) and tap **Start camera**, or
serve the folder locally:

```
python -m http.server 8000      # then open http://localhost:8000
```

A camera needs a secure context, so `https://…` (GitHub Pages) or `localhost` is
required; opening `index.html` straight from disk will not get camera access.

## What the page does to each frame

Everything comes from [`model_info.json`](model_info.json) — nothing about the model is
hardcoded in the page:

1. take the **central square** of the frame (`min(width, height)`, centred);
2. **resize** it to `input_size × input_size` (96 × 96);
3. **scale** the RGB pixels to `0…1` (divide by 255);
4. **standardise** each channel with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`;
5. feed a `1 × 3 × 96 × 96` float32 tensor in planar CHW order as input `image`;
6. apply **softmax** to the `logits` output and show all four probabilities live.

The small canvas next to the camera button shows the exact 96 × 96 picture the model
receives, so you can see what the network sees.

## iPhone / Safari

* Needs **iOS 16.4 or newer**: the model runs on WebAssembly SIMD, which Safari added
  there. Recent Chrome and Edge on Android and desktop work as well.
* [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) is loaded from a CDN
  (jsDelivr), pinned, with `numThreads = 1` — the single-threaded wasm build is the one
  that runs without cross-origin-isolated shared memory, which GitHub Pages does not
  provide. The page tries `1.22.0` first and falls back to `1.20.1` then `1.19.2` if a
  browser cannot run the newest WebAssembly binary. (onnxruntime-web ≤ 1.17 aborts on
  this model, so it is not offered.)
* The camera only starts from a tap (iOS autoplay policy), which is why it is behind a
  button, and the `<video>` element uses `playsinline` so it stays in the page instead
  of going fullscreen.
* Rear-camera frames arrive in the sensor's landscape orientation, so the upright
  picture can be a quarter turn away from what the model expects. The page guesses from
  `screen.orientation` and gives you a **Rotate input** button to correct it by eye —
  watch the 96 × 96 preview.

## Photo upload (fallback)

**Upload photo** runs the identical preprocessing on any still image and needs no camera
permission. `check_image.jpg` is a known test image; uploading it must reproduce
[`check_expected.json`](check_expected.json):

| class | expected | this page (Chromium and WebKit) |
|---|---|---|
| apple | 0.8784665 | 0.8784680 |
| cherry | 0.1206345 | 0.1206329 |
| chestnut | 0.0006078 | 0.0006078 |
| maple | 0.0002912 | 0.0002912 |

Largest deviation 1.5 × 10⁻⁶, which is JPEG-decoder and float32 rounding. The same
numbers come out of an independent reference run in Python (ONNX Runtime + Pillow), and
the table above was measured in the deployed page under both Chromium and WebKit.

## Files

| file | purpose |
|---|---|
| `index.html` | the whole app: camera loop, preprocessing, inference, results, QR encoder |
| `leaf_model.onnx` | the trained classifier (input `image` `1×3×96×96` float32, output `logits`) |
| `model_info.json` | class names, input size, per-channel mean and std |
| `check_image.jpg`, `check_expected.json` | self-check pair for the upload path |

## Known limits

* Class names are the four from `model_info.json`; the model was not trained to reject
  non-leaf input, so it always answers with one of them.
* A photo taken through the iOS shutter carries EXIF orientation and is handled; the
  live stream is not automatically rotated for the reasons above.
* Inference is single-threaded WebAssembly, paced to at most four frames per second to
  keep the battery and fan quiet. Measured inference time was 4–14 ms per frame on a
  desktop CPU; it is not measured on a phone from here, so expect it to be slower there.
