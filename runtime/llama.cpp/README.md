# SenseVoiceSmall on llama.cpp / GGUF

Run **SenseVoiceSmall** on the [llama.cpp](https://github.com/ggml-org/llama.cpp) /
ggml stack — CPU, edge, single binary, no Python at runtime.

SenseVoiceSmall = SAN-M encoder + CTC head (no LLM). The pipeline runs in C++:

```
WAV (16k mono) → kaldi fbank → prepend 4 query tokens → SAN-M encoder (ggml) →
CTC head → greedy CTC decode → token ids → SentencePiece detok → text
```

## Contents
- `funasr-sensevoice/` — the ggml runtime: WAV → CTC token ids
- `export_sensevoice_gguf.py` — export encoder + CTC head + query embeddings to GGUF
- `detok.py` — SentencePiece id → text (the bpe model ships with the checkpoint)

## Build
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/funasr-sensevoice examples/
echo 'add_subdirectory(funasr-sensevoice)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-sensevoice
```

## Convert weights to GGUF (one-time)
```bash
python export_sensevoice_gguf.py \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small.gguf                 # f32
python export_sensevoice_gguf.py --wtype f16 \
    --model_pt <model>/model.pt --mvn <model>/am.mvn \
    --out sensevoice-small-f16.gguf             # half size
```

## Transcribe
```bash
build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf -a audio.wav > ids.txt
python detok.py <model>/chn_jpn_yue_eng_ko_spectok.bpe.model ids.txt
# → <|zh|><|NEUTRAL|><|Speech|><|woitn|>...transcription...
```

## Notes & validation
- The 70-layer SAN-M encoder is shared with the Fun-ASR-Nano runtime and was
  validated against PyTorch (cosine 1.0). On a benchmark clip the C++ CTC token
  ids are **identical** to PyTorch (108/108), and the detokenized text matches
  the FunASR `AutoModel` output exactly.
- Inference feeds the **raw** log-mel fbank to the encoder; SenseVoice does **not**
  apply am.mvn CMVN at inference (doing so makes it predict `<|nospeech|>`).
- Query tokens prepended (4): `[language(auto=0), event=1, emotion=2, textnorm(woitn=15)]`
  from `embed.weight`; change indices for language/ITN options.
- LayerNorm eps 1e-5; FSMN = exact f32 shift-accumulate; fbank matches torchaudio.
- WAV input currently assumes 16 kHz mono PCM16.

Requires a SenseVoiceSmall checkpoint (e.g. `FunAudioLLM/SenseVoiceSmall`).
