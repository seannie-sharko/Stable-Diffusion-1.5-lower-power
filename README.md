# Intel N150 OpenVINO Img2Img + Real-ESRGAN + Gradio

![Alt text](/img/IMG_9969.png)

![Alt text](/img/IMG_4206.png)	

![Alt text](/img/IMG_7528.png)	
	
This guide documents a working local image-to-image setup built and tested on 
a low-power Intel N150 system using the integrated Intel GPU.

![Alt text](/img/Image_1.jpg)	


The setup uses:

- **Linux Mint 22.1**
- **Intel N150**
- **Intel integrated graphics**
- **12 GiB RAM**
- **OpenVINO**
- **Optimum Intel**
- **Stable Diffusion 1.5 / Realistic Vision V6.0 B1**
- **Real-ESRGAN NCNN Vulkan**
- **Gradio**
- **systemd user services**

The final design keeps the Stable Diffusion backend loaded in memory so it does 
not reload the model for every generation.

---

## 1. Final architecture

```text
Browser
  |
  v
Gradio UI
0.0.0.0:7860
$HOME/openvino-img2img-ui/.venv
  |
  | localhost TCP
  v
Persistent backend
127.0.0.1:8765
$HOME/openvino-img2img/.venv
  |
  +-- Realistic Vision / OpenVINO
  +-- Intel iGPU
  +-- Real-ESRGAN / Vulkan
```

The backend loads Realistic Vision once and remains resident.

The Gradio frontend communicates with it over a small localhost TCP/JSON protocol.

---

## 2. Hardware used

```text
CPU: Intel N150
CPU cores: 4
RAM: 12 GiB
GPU: Intel Alder Lake-N / ADL-N integrated graphics
PCI ID: 8086:46d4
```

This is modest hardware, so this guide deliberately uses Stable Diffusion 1.5 
rather than SDXL, FLUX, or other significantly heavier models.

---

## 3. Upgrade to a kernel that supports the Intel GPU correctly

The original kernel did not expose `/dev/dri`.

Check:

```bash
ls -l /dev/dri
```

If `/dev/dri` is missing, install the current HWE kernel.

Upgrade the HWE Linux Mint / Ubuntu 24.04 base:

```bash
sudo apt update
sudo apt install linux-generic-hwe-24.04
```

Reboot:

```bash
sudo reboot
```

After reboot:

```bash
lspci -nnk -s 00:02.0
```

Expected:

```text
Kernel driver in use: i915
Kernel modules: i915, xe
```

Then:

```bash
ls -l /dev/dri
```

Expected entries include:

```text
card0
renderD128
```

---

## 4. Make sure your user account can access the GPU

Add your user to the `video` and `render` groups:

```bash
sudo usermod -aG video,render "$USER"
```

Log out and back in, or reboot.

Verify:

```bash
groups
```

You should see:

```text
video
render
```

---

## 5. Install the newer Intel compute stack

The stock Ubuntu Noble Intel compute packages were too old for this setup.

Add the Intel graphics PPA:

```bash
sudo apt install -y software-properties-common

sudo add-apt-repository -y \
  ppa:kobuk-team/intel-graphics

sudo apt update
```

Then install or upgrade the Intel compute packages:

```bash
sudo apt install -y \
  intel-opencl-icd \
  libze-intel-gpu1 \
  libze1 \
  clinfo
```

Reboot:

```bash
sudo reboot
```

Verify OpenCL:

```bash
clinfo -l
```

The system should return something like:

```text
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Graphics
```

More detailed confirmation:

```bash
clinfo | grep -E \
'Device Version|Driver Version'
```

The system should return something like:

```text
Device Version OpenCL 3.0 NEO
```

---

## 6. Verify Level Zero

Check for the Intel Level Zero library:

```bash
ls -l \
/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*
```

The tested setup had:

```text
libze_intel_gpu.so.1
```

and the versioned library beside it.

---

## 7. Verify Vulkan

Real-ESRGAN uses Vulkan.

Install Vulkan tools if needed:

```bash
sudo apt install -y vulkan-tools
```

Then:

```bash
vulkaninfo
```

On a headless machine, warnings such as this are normal:

```text
DISPLAY not set
Skipping surface info
```

The important part is that the Intel GPU appears as a Vulkan device.

The tested setup also showed `llvmpipe`, but the Intel GPU was used explicitly with:

```text
-g 0
```

---

# OpenVINO image-to-image backend

## 8. Create the backend project

```bash
mkdir -p "$HOME/openvino-img2img"

cd "$HOME/openvino-img2img"

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
```

Install the main packages:

```bash
pip install \
  "optimum[openvino]" \
  diffusers \
  transformers \
  accelerate \
  pillow \
  safetensors
```

During setup, package compatibility mattered.

The working package combination was:

```text
optimum-intel     2.2.0
optimum           2.3.0
diffusers         0.35.1
transformers      4.57.6
openvino          2026.4.1
huggingface-hub   0.36.2
```

To reproduce the same known-good combination:

```bash
pip install \
  "optimum-intel==2.2.0" \
  "optimum==2.3.0" \
  "diffusers==0.35.1" \
  "transformers==4.57.6" \
  "huggingface-hub==0.36.2"
```

Check:

```bash
pip check
```

You want:

```text
No broken requirements found.
```

Verify imports:

```bash
python - <<'PY'
from optimum.intel import OVStableDiffusionImg2ImgPipeline
import transformers
import huggingface_hub

print("transformers:", transformers.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("OpenVINO img2img: OK")
PY
```

---

## 9. Model

The model that worked well on this hardware:

```text
SG161222/Realistic_Vision_V6.0_B1_noVAE
```

The converted OpenVINO model is stored at:

```text
$HOME/openvino-img2img/models/realistic-vision-openvino
```

The first run downloads and converts the model.

Later runs load the converted model directly.

---

# Real-ESRGAN

## 10. Install Real-ESRGAN NCNN Vulkan

Use the Linux Real-ESRGAN NCNN Vulkan release.

The tested executable ended up here:

```text
$HOME/tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan
```

The model files ended up here:

```text
$HOME/tools/models/
```

The important model files are:

```text
realesrgan-x4plus.param
realesrgan-x4plus.bin
```

Verify:

```bash
ls -l \
"$HOME/tools/models/realesrgan-x4plus.param" \
"$HOME/tools/models/realesrgan-x4plus.bin"
```

---

## 11. The Real-ESRGAN settings that actually worked

A very important discovery was that this combination:

```text
model: realesrgan-x4plus
scale: 4
tile: 0
GPU: 0
```

worked correctly.

This command eliminated a "sliding picture puzzle" artifact that appeared when using direct 2x output and forced tiling:

```bash
"$HOME/tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan" \
  -i output/refined.png \
  -o output/realesrgan-x4.png \
  -n realesrgan-x4plus \
  -s 4 \
  -t 0 \
  -g 0 \
  -m "$HOME/tools/models"
```

Recommended defaults:

```text
-s 4
-t 0
-g 0
```

Do not default to:

```text
-s 2
-t 128
```

on this system.

The x4 model behaved better at its native x4 scale with automatic tile selection.

---

# Standalone img2img script

## 12. Recommended generation settings

For portrait-oriented img2img, the final recommended defaults were:

```text
First pass:
  Steps:     28
  CFG:       6.0
  Strength:  0.25

Refinement:
  Steps:     32
  CFG:       6.0
  Strength:  0.15

Real-ESRGAN:
  Scale:     4
  Tile:      0
  GPU:       0
```

Lower denoise strengths helped preserve facial structure much better than the earlier:

```text
0.45 first pass
0.30 refinement
```

---

## 13. Suggested prompt

```text
realistic portrait photograph, preserve facial structure,
preserve identity, natural eyes, detailed eyes,
natural nose, natural lips, realistic skin texture,
detailed hair, soft natural lighting, sharp focus
```

Suggested negative prompt:

```text
deformed face, distorted face, asymmetrical face,
missing eyes, malformed eyes, crossed eyes,
deformed nose, deformed mouth, bad anatomy,
blurry, low quality, waxy skin, plastic skin,
artifacts, extra features, watermark, text
```

Note that prompt phrases such as `preserve identity` can encourage the model, but plain SD 1.5 img2img does not provide true identity locking.

The biggest preservation control is still `strength`.

---

## 14. Base image sizes

The working preprocessing sizes were:

```python
PORTRAIT_BOX = (512, 768)
LANDSCAPE_BOX = (768, 512)
SQUARE_BOX = (512, 512)
```

Images are resized while preserving aspect ratio and then rounded down to a multiple of 8.

This keeps memory requirements manageable on the Intel N150.

---

# Persistent backend

## 15. Why use a persistent backend

The original frontend started a new Python process for every generation.

That meant:

```text
click Generate
  |
  +-- start Python
  +-- load OpenVINO model
  +-- generate
  +-- exit
```

The improved design does this:

```text
system boot
  |
  +-- backend starts
  +-- OpenVINO model loads once
  |
  +-- Generate
  +-- Generate
  +-- Generate
```

This is much faster between jobs.

---

## 16. `backend_server.py`

Save as:

```text
$HOME/openvino-img2img/backend_server.py
```

```python
import json
import socketserver
import subprocess
import traceback
import uuid
from pathlib import Path

from PIL import Image, ImageOps
from optimum.intel import (
    OVStableDiffusionImg2ImgPipeline,
)


HOST = "127.0.0.1"
PORT = 8765


BASE_DIR = (
    Path.home()
    / "openvino-img2img"
)

MODEL_ID = (
    "SG161222/"
    "Realistic_Vision_V6.0_B1_noVAE"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "realistic-vision-openvino"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "output"
    / "gradio_runs"
)


DEFAULT_PROMPT = (
    "realistic portrait photograph, preserve facial structure, "
    "preserve identity, natural eyes, detailed eyes, "
    "natural nose, natural lips, realistic skin texture, "
    "detailed hair, soft natural lighting, sharp focus"
)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed face, distorted face, asymmetrical face, "
    "missing eyes, malformed eyes, crossed eyes, "
    "deformed nose, deformed mouth, bad anatomy, "
    "blurry, low quality, waxy skin, plastic skin, "
    "artifacts, extra features, watermark, text"
)


DEFAULT_FIRST_STEPS = 28
DEFAULT_FIRST_GUIDANCE = 6.0
DEFAULT_FIRST_STRENGTH = 0.25

DEFAULT_SECOND_STEPS = 32
DEFAULT_SECOND_GUIDANCE = 6.0
DEFAULT_SECOND_STRENGTH = 0.15


PORTRAIT_BOX = (512, 768)
LANDSCAPE_BOX = (768, 512)
SQUARE_BOX = (512, 512)


REALESRGAN_BIN = (
    Path.home()
    / "tools"
    / "realesrgan-ncnn-vulkan"
    / "realesrgan-ncnn-vulkan"
)

REALESRGAN_MODEL_DIR = (
    Path.home()
    / "tools"
    / "models"
)

REALESRGAN_MODEL = "realesrgan-x4plus"

DEFAULT_UPSCALE = 4
DEFAULT_TILE = 0
REALESRGAN_GPU = "0"


def round_down_8(value):
    return max(
        8,
        value - (value % 8),
    )


def prepare_image(image):
    width, height = image.size

    if width == height:
        box = SQUARE_BOX
    elif height > width:
        box = PORTRAIT_BOX
    else:
        box = LANDSCAPE_BOX

    image = ImageOps.contain(
        image,
        box,
        method=Image.Resampling.LANCZOS,
    )

    new_width = round_down_8(image.width)
    new_height = round_down_8(image.height)

    if image.size != (
        new_width,
        new_height,
    ):
        image = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

    return image


def lanczos_upscale(
    image,
    scale,
):
    width = round_down_8(
        image.width * scale
    )

    height = round_down_8(
        image.height * scale
    )

    return image.resize(
        (
            width,
            height,
        ),
        Image.Resampling.LANCZOS,
    )


def validate_realesrgan():
    if not REALESRGAN_BIN.is_file():
        raise FileNotFoundError(
            f"Missing executable: "
            f"{REALESRGAN_BIN}"
        )

    param_file = (
        REALESRGAN_MODEL_DIR
        / f"{REALESRGAN_MODEL}.param"
    )

    bin_file = (
        REALESRGAN_MODEL_DIR
        / f"{REALESRGAN_MODEL}.bin"
    )

    if not param_file.is_file():
        raise FileNotFoundError(
            f"Missing model: {param_file}"
        )

    if not bin_file.is_file():
        raise FileNotFoundError(
            f"Missing model: {bin_file}"
        )


def run_realesrgan(
    input_file,
    output_file,
    scale,
    tile,
):
    validate_realesrgan()

    cmd = [
        str(REALESRGAN_BIN),
        "-i",
        str(input_file),
        "-o",
        str(output_file),
        "-n",
        REALESRGAN_MODEL,
        "-s",
        str(scale),
        "-t",
        str(tile),
        "-g",
        REALESRGAN_GPU,
        "-m",
        str(REALESRGAN_MODEL_DIR),
        "-f",
        "png",
    ]

    print()
    print("Running Real-ESRGAN...")
    print(" ".join(cmd))
    print()

    subprocess.run(
        cmd,
        check=True,
    )


def load_pipeline():
    print(
        "Loading Realistic Vision "
        "into OpenVINO..."
    )

    if MODEL_DIR.exists():
        pipe = (
            OVStableDiffusionImg2ImgPipeline
            .from_pretrained(
                MODEL_DIR,
                device="GPU",
                safety_checker=None,
            )
        )
    else:
        pipe = (
            OVStableDiffusionImg2ImgPipeline
            .from_pretrained(
                MODEL_ID,
                export=True,
                device="GPU",
                safety_checker=None,
            )
        )

        pipe.save_pretrained(
            MODEL_DIR
        )

    print("Model loaded.")
    print("Backend ready.")

    return pipe


PIPE = load_pipeline()


def generate(data):
    input_file = Path(
        data["input"]
    )

    if not input_file.is_file():
        raise FileNotFoundError(
            str(input_file)
        )

    run_id = uuid.uuid4().hex

    run_dir = (
        OUTPUT_ROOT
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_file = (
        run_dir
        / "first_pass.png"
    )

    refined_file = (
        run_dir
        / "refined.png"
    )

    final_file = (
        run_dir
        / "result.png"
    )

    image = Image.open(
        input_file
    ).convert("RGB")

    image = prepare_image(
        image
    )

    print()
    print(
        "Prepared input resolution:",
        f"{image.width}x{image.height}",
    )

    prompt = data.get(
        "prompt",
        DEFAULT_PROMPT,
    )

    negative_prompt = data.get(
        "negative_prompt",
        DEFAULT_NEGATIVE_PROMPT,
    )

    first_steps = int(
        data.get(
            "first_steps",
            DEFAULT_FIRST_STEPS,
        )
    )

    first_guidance = float(
        data.get(
            "first_guidance",
            DEFAULT_FIRST_GUIDANCE,
        )
    )

    first_strength = float(
        data.get(
            "first_strength",
            DEFAULT_FIRST_STRENGTH,
        )
    )

    print()
    print("Running first pass...")

    first = PIPE(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=first_strength,
        guidance_scale=first_guidance,
        num_inference_steps=first_steps,
    ).images[0]

    first.save(
        first_file
    )

    print(
        "Saved:",
        first_file,
    )

    do_refine = bool(
        data.get(
            "do_refine",
            True,
        )
    )

    if do_refine:
        second_steps = int(
            data.get(
                "second_steps",
                DEFAULT_SECOND_STEPS,
            )
        )

        second_guidance = float(
            data.get(
                "second_guidance",
                DEFAULT_SECOND_GUIDANCE,
            )
        )

        second_strength = float(
            data.get(
                "second_strength",
                DEFAULT_SECOND_STRENGTH,
            )
        )

        print()
        print("Running refinement pass...")

        refined = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=first,
            strength=second_strength,
            guidance_scale=second_guidance,
            num_inference_steps=second_steps,
        ).images[0]

    else:
        refined = first

    refined.save(
        refined_file
    )

    print(
        "Saved:",
        refined_file,
    )

    use_realesrgan = bool(
        data.get(
            "use_realesrgan",
            True,
        )
    )

    scale = int(
        data.get(
            "upscale",
            DEFAULT_UPSCALE,
        )
    )

    tile = int(
        data.get(
            "tile",
            DEFAULT_TILE,
        )
    )

    if use_realesrgan:
        try:
            run_realesrgan(
                refined_file,
                final_file,
                scale,
                tile,
            )

        except Exception as error:
            print()
            print("Real-ESRGAN failed:")
            print(error)

            final = lanczos_upscale(
                refined,
                scale,
            )

            final.save(
                final_file
            )

    else:
        final = lanczos_upscale(
            refined,
            scale,
        )

        final.save(
            final_file
        )

    final = Image.open(
        final_file
    )

    print()
    print(
        "Final resolution:",
        f"{final.width}x{final.height}",
    )

    print(
        "Saved:",
        final_file,
    )

    return {
        "ok": True,
        "run_id": run_id,
        "first": str(first_file),
        "refined": str(refined_file),
        "final": str(final_file),
        "width": final.width,
        "height": final.height,
        "first_steps": first_steps,
        "first_guidance": first_guidance,
        "first_strength": first_strength,
        "refine_used": do_refine,
        "realesrgan_used": use_realesrgan,
        "upscale": scale,
        "tile": tile,
    }


class RequestHandler(
    socketserver.StreamRequestHandler
):
    def handle(self):
        try:
            raw = self.rfile.readline()

            request = json.loads(
                raw.decode("utf-8")
            )

            action = request.get(
                "action"
            )

            if action == "ping":
                response = {
                    "ok": True,
                    "status": "ready",
                }

            elif action == "generate":
                response = generate(
                    request
                )

            else:
                response = {
                    "ok": False,
                    "error": (
                        f"Unknown action: "
                        f"{action}"
                    ),
                }

        except Exception:
            response = {
                "ok": False,
                "error": traceback.format_exc(),
            }

        payload = (
            json.dumps(response)
            + "\n"
        )

        self.wfile.write(
            payload.encode("utf-8")
        )


class BackendServer(
    socketserver.TCPServer
):
    allow_reuse_address = True


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


if __name__ == "__main__":
    print()
    print(
        f"Listening on "
        f"{HOST}:{PORT}"
    )

    with BackendServer(
        (
            HOST,
            PORT,
        ),
        RequestHandler,
    ) as server:
        server.serve_forever()
```

Check syntax:

```bash
cd "$HOME/openvino-img2img"
source .venv/bin/activate

python -m py_compile backend_server.py
```

---

# Gradio frontend

## 17. Keep the frontend in a separate venv

Do **not** install Gradio into the OpenVINO backend environment.

Gradio previously upgraded `huggingface-hub` to a version incompatible with the OpenVINO/Transformers stack.

Instead:

```bash
mkdir -p "$HOME/openvino-img2img-ui"

cd "$HOME/openvino-img2img-ui"

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip
```

Create:

```text
requirements.txt
```

with:

```text
gradio==6.29.1
Pillow==12.3.0
```

Install:

```bash
pip install -r requirements.txt

pip check
```

---

## 18. `app.py`

Save as:

```text
$HOME/openvino-img2img-ui/app.py
```

```python
import json
import socket
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image


BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8765


BACKEND_DIR = (
    Path.home()
    / "openvino-img2img"
)

UPLOAD_DIR = (
    BACKEND_DIR
    / "input"
    / "gradio"
)

ALLOWED_OUTPUT_DIR = (
    BACKEND_DIR
    / "output"
    / "gradio_runs"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DEFAULT_PROMPT = (
    "realistic portrait photograph, preserve facial structure, "
    "preserve identity, natural eyes, detailed eyes, "
    "natural nose, natural lips, realistic skin texture, "
    "detailed hair, soft natural lighting, sharp focus"
)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed face, distorted face, asymmetrical face, "
    "missing eyes, malformed eyes, crossed eyes, "
    "deformed nose, deformed mouth, bad anatomy, "
    "blurry, low quality, waxy skin, plastic skin, "
    "artifacts, extra features, watermark, text"
)


DEFAULT_FIRST_STEPS = 28
DEFAULT_FIRST_CFG = 6.0
DEFAULT_FIRST_STRENGTH = 0.25

DEFAULT_SECOND_STEPS = 32
DEFAULT_SECOND_CFG = 6.0
DEFAULT_SECOND_STRENGTH = 0.15

DEFAULT_UPSCALE = 4
DEFAULT_TILE = 0


def backend_request(request):
    data = (
        json.dumps(request)
        + "\n"
    ).encode("utf-8")

    with socket.create_connection(
        (
            BACKEND_HOST,
            BACKEND_PORT,
        ),
        timeout=10,
    ) as sock:

        sock.settimeout(None)
        sock.sendall(data)

        response = b""

        while not response.endswith(b"\n"):
            chunk = sock.recv(65536)

            if not chunk:
                break

            response += chunk

    if not response:
        raise RuntimeError(
            "Backend returned no response."
        )

    return json.loads(
        response.decode("utf-8")
    )


def check_backend():
    try:
        response = backend_request(
            {
                "action": "ping",
            }
        )

        if response.get("ok"):
            return "Backend: READY"

        return (
            "Backend error: "
            + str(response)
        )

    except Exception as error:
        return (
            "Backend unavailable: "
            + str(error)
        )


def generate(
    input_image,
    prompt,
    negative_prompt,
    first_steps,
    first_cfg,
    first_strength,
    do_refine,
    second_steps,
    second_cfg,
    second_strength,
    use_realesrgan,
    upscale,
    tile,
):
    if input_image is None:
        raise gr.Error(
            "Please upload an image."
        )

    if not prompt or not prompt.strip():
        raise gr.Error(
            "Please enter a prompt."
        )

    filename = (
        f"{int(time.time())}-"
        f"{uuid.uuid4().hex[:8]}.png"
    )

    input_file = (
        UPLOAD_DIR
        / filename
    )

    if isinstance(
        input_image,
        Image.Image,
    ):
        image = input_image
    else:
        image = Image.fromarray(
            input_image
        )

    image = image.convert("RGB")

    image.save(
        input_file
    )

    request = {
        "action": "generate",
        "input": str(input_file),
        "prompt": prompt,
        "negative_prompt": (
            negative_prompt or ""
        ),
        "first_steps": int(
            first_steps
        ),
        "first_guidance": float(
            first_cfg
        ),
        "first_strength": float(
            first_strength
        ),
        "do_refine": bool(
            do_refine
        ),
        "second_steps": int(
            second_steps
        ),
        "second_guidance": float(
            second_cfg
        ),
        "second_strength": float(
            second_strength
        ),
        "use_realesrgan": bool(
            use_realesrgan
        ),
        "upscale": int(
            upscale
        ),
        "tile": int(
            tile
        ),
    }

    try:
        response = backend_request(
            request
        )

    except Exception as error:
        raise gr.Error(
            "Could not communicate with backend:\n"
            f"{error}"
        )

    if not response.get("ok"):
        raise gr.Error(
            response.get(
                "error",
                "Unknown backend error",
            )
        )

    status_lines = [
        "Finished",
        f"Run ID: {response['run_id']}",
        (
            "Final resolution: "
            f"{response['width']}x"
            f"{response['height']}"
        ),
        (
            "First pass: "
            f"{int(first_steps)} steps, "
            f"CFG {float(first_cfg):.1f}, "
            f"strength "
            f"{float(first_strength):.2f}"
        ),
    ]

    if do_refine:
        status_lines.append(
            (
                "Refinement: "
                f"{int(second_steps)} steps, "
                f"CFG {float(second_cfg):.1f}, "
                f"strength "
                f"{float(second_strength):.2f}"
            )
        )
    else:
        status_lines.append(
            "Refinement: disabled"
        )

    if use_realesrgan:
        status_lines.append(
            (
                "Real-ESRGAN: "
                f"{int(upscale)}x, "
                f"tile {int(tile)}"
            )
        )
    else:
        status_lines.append(
            (
                "Real-ESRGAN: disabled; "
                f"Lanczos {int(upscale)}x"
            )
        )

    status = "\n".join(
        status_lines
    )

    return (
        response["first"],
        response["refined"],
        response["final"],
        response["final"],
        status,
    )


with gr.Blocks(
    title="Realistic Vision Img2Img",
) as demo:

    gr.Markdown(
        "# Realistic Vision Img2Img"
    )

    gr.Markdown(
        "Realistic Vision + OpenVINO on Intel GPU "
        "+ Real-ESRGAN"
    )

    with gr.Row():

        backend_status = gr.Textbox(
            label="Backend status",
            value=check_backend(),
            interactive=False,
        )

        refresh_button = gr.Button(
            "Check Backend"
        )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                label="Input image",
                type="pil",
            )

            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_PROMPT,
            )

            negative_prompt = gr.Textbox(
                label="Negative prompt",
                lines=4,
                value=DEFAULT_NEGATIVE_PROMPT,
            )

            with gr.Accordion(
                "First Pass",
                open=True,
            ):

                first_steps = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=DEFAULT_FIRST_STEPS,
                    step=1,
                    label="Steps",
                )

                first_cfg = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=DEFAULT_FIRST_CFG,
                    step=0.1,
                    label="CFG",
                )

                first_strength = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=DEFAULT_FIRST_STRENGTH,
                    step=0.01,
                    label="Strength",
                )

            with gr.Accordion(
                "Refinement",
                open=True,
            ):

                do_refine = gr.Checkbox(
                    value=True,
                    label="Enable refinement",
                )

                second_steps = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=DEFAULT_SECOND_STEPS,
                    step=1,
                    label="Steps",
                )

                second_cfg = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=DEFAULT_SECOND_CFG,
                    step=0.1,
                    label="CFG",
                )

                second_strength = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=DEFAULT_SECOND_STRENGTH,
                    step=0.01,
                    label="Strength",
                )

            with gr.Accordion(
                "Upscaling",
                open=True,
            ):

                use_realesrgan = gr.Checkbox(
                    value=True,
                    label="Use Real-ESRGAN",
                )

                upscale = gr.Radio(
                    choices=[
                        2,
                        3,
                        4,
                    ],
                    value=DEFAULT_UPSCALE,
                    label="Upscale factor",
                )

                tile = gr.Radio(
                    choices=[
                        0,
                        64,
                        128,
                        256,
                    ],
                    value=DEFAULT_TILE,
                    label="Real-ESRGAN tile size",
                )

                gr.Markdown(
                    "Recommended for this system: "
                    "**4x upscale, tile 0 (automatic)**"
                )

            generate_button = gr.Button(
                "Generate",
                variant="primary",
            )

        with gr.Column():

            first_output = gr.Image(
                label="First Pass",
                type="filepath",
            )

            refined_output = gr.Image(
                label="Refined",
                type="filepath",
            )

            final_output = gr.Image(
                label="Final",
                type="filepath",
            )

            download_output = gr.File(
                label="Download Final",
            )

            status_output = gr.Textbox(
                label="Status",
                lines=8,
                interactive=False,
            )

    refresh_button.click(
        fn=check_backend,
        inputs=[],
        outputs=[
            backend_status,
        ],
    )

    generate_button.click(
        fn=generate,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            first_steps,
            first_cfg,
            first_strength,
            do_refine,
            second_steps,
            second_cfg,
            second_strength,
            use_realesrgan,
            upscale,
            tile,
        ],
        outputs=[
            first_output,
            refined_output,
            final_output,
            download_output,
            status_output,
        ],
        concurrency_limit=1,
    )


demo.queue(
    max_size=8,
)


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[
            str(ALLOWED_OUTPUT_DIR),
        ],
    )
```

Syntax check:

```bash
cd "$HOME/openvino-img2img-ui"
source .venv/bin/activate

python -m py_compile app.py
```

---

# systemd user services

## 19. Backend service

Create:

```bash
mkdir -p "$HOME/.config/systemd/user"
```

Then:

```text
$HOME/.config/systemd/user/openvino-img2img-backend.service
```

```ini
[Unit]
Description=OpenVINO Img2Img Backend
After=network.target

[Service]
Type=simple

WorkingDirectory=%h/openvino-img2img

ExecStart=%h/openvino-img2img/.venv/bin/python \
          %h/openvino-img2img/backend_server.py

Restart=on-failure
RestartSec=5

Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

---

## 20. Gradio service

Create:

```text
$HOME/.config/systemd/user/openvino-img2img-ui.service
```

```ini
[Unit]
Description=OpenVINO Img2Img Gradio UI
After=openvino-img2img-backend.service
Wants=openvino-img2img-backend.service

[Service]
Type=simple

WorkingDirectory=%h/openvino-img2img-ui

ExecStart=%h/openvino-img2img-ui/.venv/bin/python \
          %h/openvino-img2img-ui/app.py

Restart=on-failure
RestartSec=5

Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

---

## 21. Enable and start

```bash
systemctl --user daemon-reload
```

Enable:

```bash
systemctl --user enable \
  openvino-img2img-backend.service

systemctl --user enable \
  openvino-img2img-ui.service
```

Start:

```bash
systemctl --user start \
  openvino-img2img-backend.service

systemctl --user start \
  openvino-img2img-ui.service
```

---

## 22. Start services at boot without login

Enable lingering:

```bash
sudo loginctl enable-linger "$USER"
```

Verify:

```bash
loginctl show-user "$USER" -p Linger
```

Expected:

```text
Linger=yes
```

---

# Using the UI

## 23. Open the page

From another machine on the LAN:

```text
http://<lan-hostname-or-ip>:7860
```

The UI should show:

```text
Backend: READY
```

Recommended defaults:

```text
First Pass
  Steps:       28
  CFG:         6.0
  Strength:    0.25

Refinement
  Enabled:     Yes
  Steps:       32
  CFG:         6.0
  Strength:    0.15

Real-ESRGAN
  Enabled:     Yes
  Scale:       4
  Tile:        0
```

---

# Logs and diagnostics

## 24. Backend status

```bash
systemctl --user status \
  openvino-img2img-backend.service
```

---

## 25. UI status

```bash
systemctl --user status \
  openvino-img2img-ui.service
```

---

## 26. Follow backend logs

```bash
journalctl --user \
  -u openvino-img2img-backend.service \
  -f
```

---

## 27. Follow UI logs

```bash
journalctl --user \
  -u openvino-img2img-ui.service \
  -f
```

---

# Troubleshooting

## 28. `clinfo -l` returns nothing

Check:

```bash
ls -l /dev/dri
```

If `/dev/dri` is missing, the GPU kernel driver is not available.

Upgrade to the HWE kernel, reboot, and verify `i915` is in use.

---

## 29. `/dev/dri` exists but OpenCL is still unavailable

Check:

```bash
clinfo -l
```

If the Intel OpenCL stack is too old, use the Intel graphics PPA described earlier.

---

## 30. Transformers / Diffusers compatibility problems

Two errors encountered during setup were:

```text
AttributeError:
module transformers has no attribute CLIPFeatureExtractor
```

and later:

```text
cannot import name 'Qwen3VLModel' from transformers
```

The final working combination was:

```text
diffusers        0.35.1
transformers     4.57.6
optimum-intel    2.2.0
optimum          2.3.0
huggingface-hub  0.36.2
```

Check:

```bash
pip check
```

---

## 31. Do not install Gradio in the backend venv

Installing Gradio in the same venv upgraded:

```text
huggingface-hub
```

to a version incompatible with Transformers / Optimum Intel.

If this happens:

```bash
cd "$HOME/openvino-img2img"

source .venv/bin/activate

pip uninstall -y \
  gradio \
  gradio-client \
  hf-gradio

pip install \
  "huggingface-hub==0.36.2"

pip check
```

Then keep Gradio in its own venv.

---

## 32. Gradio error: click outside Blocks context

If you see:

```text
AttributeError:
Cannot call click outside of a gradio.Blocks context
```

the event handlers:

```python
refresh_button.click(...)
generate_button.click(...)
```

must remain inside:

```python
with gr.Blocks(...) as demo:
```

The `app.py` in this README already does this correctly.

---

## 33. Gradio InvalidPathError

If you see an error similar to:

```text
Cannot move .../gradio_runs/.../result.png
to the gradio cache dir
```

make sure `demo.launch()` contains:

```python
allowed_paths=[
    str(ALLOWED_OUTPUT_DIR),
]
```

This permits Gradio to serve generated files outside the UI project's working directory.

---

## 34. Sliding-picture-puzzle artifacts

Symptom:

- final output contains large rectangular regions
- parts of the image look shifted or independently processed
- resembles a sliding tile puzzle

The configuration that caused trouble was approximately:

```text
Real-ESRGAN x2
forced tile 128
```

The configuration that fixed it was:

```text
Real-ESRGAN model: realesrgan-x4plus
scale: 4
tile: 0
GPU: 0
```

Test manually:

```bash
"$HOME/tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan" \
  -i output/refined.png \
  -o output/realesrgan-x4.png \
  -n realesrgan-x4plus \
  -s 4 \
  -t 0 \
  -g 0 \
  -m "$HOME/tools/models"
```

---

## 35. CLI works but the Gradio page still uses old settings

Restart both services after editing Python files:

```bash
systemctl --user restart \
  openvino-img2img-backend.service

systemctl --user restart \
  openvino-img2img-ui.service
```

Then hard-refresh the browser.

Remember: frontend values explicitly sent to the backend override backend defaults.

For example, if the UI sends:

```text
upscale = 2
tile = 128
```

the backend will use those even if its own defaults are:

```text
upscale = 4
tile = 0
```

---

## 36. Confirm the actual Real-ESRGAN command

Follow backend logs:

```bash
journalctl --user \
  -u openvino-img2img-backend.service \
  -f
```

During upscaling, you want to see:

```text
-s 4
-t 0
-g 0
```

For a refined input such as:

```text
512x712
```

a true 4x output should be:

```text
2048x2848
```

If the result is:

```text
1024x1424
```

the frontend is still requesting 2x.

---

## 37. Facial structure changes too much

Reduce img2img strength.

The older settings:

```text
First strength:  0.45
Second strength: 0.30
```

allowed substantial facial reconstruction.

The improved portrait-oriented settings are:

```text
First strength:  0.25
Second strength: 0.15
```

If even more preservation is needed, try:

```text
First strength:  0.20
Second strength: 0.10
```

Plain img2img cannot guarantee identity preservation.

For stronger identity or pose control, future upgrades could include:

- IP-Adapter
- ControlNet
- face-specific restoration
- dedicated identity-conditioning methods

These were not required for the working setup documented here.

---

# File layout

The final layout is approximately:

```text
$HOME/
|
+-- openvino-img2img/
|   |
|   +-- .venv/
|   |
|   +-- backend_server.py
|   +-- img2img.py
|   |
|   +-- models/
|   |   +-- realistic-vision-openvino/
|   |
|   +-- input/
|   |   +-- input.jpg
|   |   +-- gradio/
|   |
|   +-- output/
|       +-- first_pass.png
|       +-- refined.png
|       +-- result.png
|       +-- gradio_runs/
|
+-- openvino-img2img-ui/
|   |
|   +-- .venv/
|   +-- app.py
|   +-- requirements.txt
|
+-- tools/
    |
    +-- realesrgan-ncnn-vulkan/
    |   +-- realesrgan-ncnn-vulkan
    |
    +-- models/
        +-- realesrgan-x4plus.param
        +-- realesrgan-x4plus.bin
```

systemd files:

```text
$HOME/.config/systemd/user/
|
+-- openvino-img2img-backend.service
+-- openvino-img2img-ui.service
```

---

# Security notes

The backend listens only on:

```text
127.0.0.1:8765
```

which is good.

The Gradio frontend listens on:

```text
0.0.0.0:7860
```

so it is reachable by other systems on the LAN.

Do not expose port `7860` directly to the public Internet unless authentication and a secure reverse proxy or VPN are added.

---

# Known-good summary

The setup that ultimately worked was:

```text
Linux Mint 22.1
Intel N150
Intel iGPU
12 GiB RAM

Kernel:
HWE kernel with working i915 device nodes

Compute:
Intel OpenCL / Level Zero
OpenVINO

Model:
Realistic Vision V6.0 B1 noVAE

Diffusion:
First:
  28 steps
  CFG 6.0
  strength 0.25

Second:
  32 steps
  CFG 6.0
  strength 0.15

Base resolution:
up to roughly 512x768

Upscale:
Real-ESRGAN NCNN Vulkan
realesrgan-x4plus
scale 4
tile 0
GPU 0

Frontend:
Gradio 6.29.1

Architecture:
persistent OpenVINO backend
separate Gradio frontend
systemd user services
```

---

# Useful maintenance commands

Restart everything:

```bash
systemctl --user restart \
  openvino-img2img-backend.service

systemctl --user restart \
  openvino-img2img-ui.service
```

Check everything:

```bash
systemctl --user status \
  openvino-img2img-backend.service \
  --no-pager

systemctl --user status \
  openvino-img2img-ui.service \
  --no-pager
```

Backend logs:

```bash
journalctl --user \
  -u openvino-img2img-backend.service \
  -f
```

UI logs:

```bash
journalctl --user \
  -u openvino-img2img-ui.service \
  -f
```

Check Intel GPU:

```bash
clinfo -l
```

Check Vulkan:

```bash
vulkaninfo
```

Check Python dependency health:

```bash
"$HOME/openvino-img2img/.venv/bin/pip" check

"$HOME/openvino-img2img-ui/.venv/bin/pip" check
```

---

## Final note

This configuration was tuned around a low-power Intel N150 system, so the settings favor stability and modest memory use rather than maximum raw generation speed.

The two choices that made the largest practical difference were:

1. **keeping the OpenVINO model loaded in a persistent backend**
2. **running `realesrgan-x4plus` at native 4x with automatic tiling (`-s 4 -t 0`)**

Those two changes produced the most reliable version of the setup.
