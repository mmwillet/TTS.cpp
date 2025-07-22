#!/bin/bash
set -e
# Performs snapshot testing against mmwillet2's version on Hugging Face

# To avoid redownloading models, cd to a persistent directory instead of build/bin
[ $# -eq 0 ] || { echo 'Usage: ./quantize_snapshot_test.sh'; exit 1; }
quantize="$(dirname "$0")"/quantize
[ -x "$quantize" ] || { echo 'quantize binary in script directory is not executable'; exit 1; }

if [ ! -e gguf_py_venv ]; then
  read -rp 'Path to llama.cpp (or blank): ' llamacpppath
  if [ -z "$llamacpppath" ]; then
    ln -s /dev/null gguf_py_venv
  else
    [ -f "$llamacpppath"/gguf-py/pyproject.toml ] || { echo 'Incompatible llama.cpp or path'; exit 1; }
    pushd "$llamacpppath"/gguf-py
    if [ ! -d venv ]; then
      python3 -m venv venv
      (
        . venv/bin/activate
        pip install -e .
      )
    fi
    popd
    ln -s "$llamacpppath"/gguf-py/venv gguf_py_venv
  fi
fi
if [ -d gguf_py_venv ]; then
  . gguf_py_venv/bin/activate
  dumper=gguf-dump
  command -v "$dumper" >/dev/null 2>&1 || { echo 'Missing gguf-dump'; exit 1; }
fi

if [ -z "$XDG_RUNTIME_DIR" ]; then
  XDG_RUNTIME_DIR=/tmp # CI or macOS
else
  size="$(sed -n "/^tmpfs ${XDG_RUNTIME_DIR//\//\\\/}/s/.\\+size=\\([0-9]\\+\\)k.\\+/\\1/p" /proc/mounts)"
  if [ -n "$size" ] && [ "$size" -lt 4194304  ]; then
    (
      set -x
      sudo mount -o remount,size=4G "$XDG_RUNTIME_DIR"
    )
  fi
fi

[ -d Dia_GGUF ] || git clone https://huggingface.co/mmwillet2/Dia_GGUF
[ -d Kokoro_GGUF ] || git clone https://huggingface.co/mmwillet2/Kokoro_GGUF
[ -d parler-tts-mini-v1-GGUF ] || git clone https://huggingface.co/ecyht2/parler-tts-mini-v1-GGUF

declare -a extra_args
function q {
  model_dir="$(dirname "$model")"
  log="$(
    set -x
    "$quantize" -mp "$model" -qt "$1" -qp "$XDG_RUNTIME_DIR"/test.gguf "${extra_args[@]}" 2>&1
  )" || echo -n "$log"

  new_hash="$(sha256sum "$XDG_RUNTIME_DIR"/test.gguf)"
  new_hash="${new_hash% *}"
  echo "$new_hash"' '"$XDG_RUNTIME_DIR"/test.gguf
  old_hash="$(git -C "$model_dir" cat-file -p HEAD:"$2" | sed -n '2s/^oid sha256://p')"
  echo "$old_hash"'  '"$model_dir"/"$2"

  [ "$new_hash" != "$old_hash" ] && [ -n "$dumper" ] && "$dumper" "$XDG_RUNTIME_DIR"/test.gguf > "$XDG_RUNTIME_DIR"/test.gguf.gguf-dump.txt 2>/dev/null
  unlink "$XDG_RUNTIME_DIR"/test.gguf
  if [ "$new_hash" != "$old_hash" ] && [ -n "$dumper" ]; then
    [ -f "$model_dir"/"$2".gguf-dump.txt ] || "$dumper" "$model_dir"/"$2" > "$model_dir"/"$2".gguf-dump.txt 2>/dev/null
    diff -U3 "$model_dir"/"$2".gguf-dump.txt "$XDG_RUNTIME_DIR"/test.gguf.gguf-dump.txt || :
  fi
}

model=Dia_GGUF/Dia.gguf
extra_args=(-nt 3)
q F16 Dia_F16.gguf
q Q4 Dia_Q4.gguf
q Q5 Dia_Q5.gguf
q Q8 Dia_Q8.gguf
extra_args=(-nt 3 -df)
q F16 Dia_F16_DAC_F16.gguf
q Q4 Dia_Q4_DAC_F16.gguf
q Q5 Dia_Q5_DAC_F16.gguf
q Q8 Dia_Q8_DAC_F16.gguf

model=Kokoro_GGUF/Kokoro_espeak.gguf
extra_args=(-nt 3 -nqf)
q F16 Kokoro_espeak_F16.gguf
q Q4 Kokoro_espeak_Q4.gguf
q Q5 Kokoro_espeak_Q5.gguf
q Q8 Kokoro_espeak_Q8.gguf
model=Kokoro_GGUF/Kokoro_no_espeak.gguf
q F16 Kokoro_no_espeak_F16.gguf
q Q4 Kokoro_no_espeak_Q4.gguf
q Q5 Kokoro_no_espeak_Q5.gguf
q Q8 Kokoro_no_espeak_Q8.gguf

model=parler-tts-mini-v1-GGUF/parler-tts-mini-v1-fp32.gguf
extra_args=(-nt 3)
q FP16 parler-tts-mini-v1-fp16.gguf
q Q4_0 parler-tts-mini-v1-Q4_0.gguf
q Q5_0 parler-tts-mini-v1-Q5_0.gguf
q Q8_0 parler-tts-mini-v1-Q8_0.gguf

rm "$XDG_RUNTIME_DIR"/test.gguf*
