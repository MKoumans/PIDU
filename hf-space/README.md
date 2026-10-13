---
title: PIDU live demo
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.13"
app_file: app.py
license: mit
models:
  - MKoumans/pidu-optical-parameter-extraction
datasets:
  - MKoumans/pidu-data
short_description: Live PIDU inference on simulated optical spectra
---

# PIDU live demo

Gradio app for [Physics-infused Deep Unfolding](https://github.com/MKoumans/PIDU): pick a simulated
test spectrum and the released PIDU checkpoint fits it in three unfolding stages, on CPU.

This folder is the source of the Hugging Face Space. To deploy, create a Gradio Space (e.g.
`MKoumans/pidu-demo`) and push the three files in this folder to it:

```bash
git clone https://huggingface.co/spaces/MKoumans/pidu-demo
cp hf-space/{app.py,requirements.txt,README.md} pidu-demo/
cd pidu-demo && git add . && git commit -m "PIDU demo" && git push
```

`requirements.txt` installs `pidu` from GitHub, so the repository must be public first. Checkpoints
and test sets are downloaded from the Hub on start-up. Run locally from the repository root:

```bash
uv run --with gradio python hf-space/app.py
```
