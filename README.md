([简体中文](./README_zh.md)|English)


# Introduction

SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR),  spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED). 

<img src="image/sensevoice2.png">

[//]: # (<div align="center"><img src="image/sensevoice.png" width="700"/> </div>)

<div align="center">  
<h4>
<a href="https://www.modelscope.cn/studios/iic/SenseVoice"> Online Demo </a>
｜<a href="# "> Homepage </a>
｜<a href="#What's News"> What's News </a>
｜<a href="#Benchmarks"> Benchmarks </a>
｜<a href="#Install"> Install </a>
｜<a href="#Usage"> Usage </a>
｜<a href="#Community"> Community </a>
</h4>

Model Zoo:
[modelscope](https://www.modelscope.cn/models/iic/SenseVoiceSmall), [huggingface](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

</div>


<a name="Highligts"></a>
# Highligts 🎯
**SenseVoice** focuses on high-accuracy multilingual speech recognition, speech emotion recognition, and audio event detection.
- **Multilingual Speech Recognition:** Trained with over 400,000 hours of data, supporting more than 50 languages, the recognition performance surpasses that of the Whisper model.
- **Rich transcribe:** 
  - Possess excellent emotion recognition capabilities, achieving and surpassing the effectiveness of the current best emotion recognition models on test data.
  - Offer sound event detection capabilities, supporting the detection of various common human-computer interaction events such as bgm, applause, laughter, crying, coughing, and sneezing.
- **Efficient Inference:** The SenseVoice-Small model utilizes a non-autoregressive end-to-end framework, leading to exceptionally low inference latency. It requires only 70ms to process 10 seconds of audio, which is 15 times faster than Whisper-Large.
- **Convenient Finetuning:** Provide convenient finetuning scripts and strategies, allowing users to easily address long-tail sample issues according to their business scenarios.
- **Service Deployment:** Offer service deployment pipeline,  supporting multi-concurrent requests, with client-side languages including Python, C++, HTML, Java, and C#, among others.

<a name="What's News"></a>
# What's News 🔥
- 2024/7: The [SenseVoice-Small](https://www.modelscope.cn/models/iic/SenseVoiceSmall) voice understanding model is open-sourced, which offers high-precision multilingual speech recognition, emotion recognition, and audio event detection capabilities for Mandarin, Cantonese, English, Japanese, and Korean and leads to exceptionally low inference latency.  
- 2024/7: The CosyVoice for natural speech generation with multi-language, timbre, and emotion control. CosyVoice excels in multi-lingual voice generation, zero-shot voice generation, cross-lingual voice cloning, and instruction-following capabilities. [CosyVoice repo](https://https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice-300M).
- 2024/7: [FunASR](https://github.com/modelscope/FunASR) is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR.

<a name="Benchmarks"></a>
# Benchmarks 📝

## Multilingual Speech Recognition
We compared the performance of multilingual speech recognition between SenseVoice and Whisper on open-source benchmark datasets, including AISHELL-1, AISHELL-2, Wenetspeech, LibriSpeech, and Common Voice. n terms of Chinese and Cantonese recognition, the SenseVoice-Small model has advantages.

<div align="center">  
<img src="image/asr_results.png" width="1000" />
</div>

## Speech Emotion Recognition

Due to the current lack of widely-used benchmarks and methods for speech emotion recognition, we conducted evaluations across various metrics on multiple test sets and performed a comprehensive comparison with numerous results from recent benchmarks. The selected test sets encompass data in both Chinese and English, and include multiple styles such as performances, films, and natural conversations. Without finetuning on the target data, SenseVoice was able to achieve and exceed the performance of the current best speech emotion recognition models.

<div align="center">  
<img src="image/ser_table.png" width="1000" />
</div>

Furthermore, we compared multiple open-source speech emotion recognition models on the test sets, and the results indicate that the SenseVoice-Large model achieved the best performance on nearly all datasets, while the SenseVoice-Small model also surpassed other open-source models on the majority of the datasets.

<div align="center">  
<img src="image/ser_figure.png" width="500" />
</div>

## Audio Event Detection

Although trained exclusively on speech data, SenseVoice can still function as a standalone event detection model. We compared its performance on the environmental sound classification ESC-50 dataset against the widely used industry models BEATS and PANN. The SenseVoice model achieved commendable results on these tasks. However, due to limitations in training data and methodology, its event classification performance has some gaps compared to specialized AED models.

<div align="center">  
<img src="image/aed_figure.png" width="500" />
</div>

## Computational  Efficiency

The SenseVoice-Small model  non-autoregressive end-to-end architecture, resulting in extremely low inference latency. With a similar number of parameters to the Whisper-Small model, it infers more than 5 times faster than Whisper-Small and 15 times faster than Whisper-Large. 

<div align="center">  
<img src="image/inference.png" width="1000" />
</div>


# Requirements

```shell
pip install -r requirements.txt
```

<a name="Usage"></a>
# Usage

## Inference

### Method 1

```python
from model import SenseVoiceSmall

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir)


res = m.inference(
    data_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    **kwargs,
)

print(res)
```

### Method 2

```python
from funasr import AutoModel

model_dir = "iic/SenseVoiceSmall"
input_file = (
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
)

model = AutoModel(model=model_dir,
                  vad_model="fsmn-vad",
                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True, device="cuda:0")

res = model.generate(
    input=input_file,
    cache={},
    language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    batch_size_s=0, 
)

print(res)
```

The funasr version has integrated the VAD (Voice Activity Detection) model and supports audio input of any duration, with `batch_size_s` in seconds.
If all inputs are short audios, and batch inference is needed to speed up inference efficiency, the VAD model can be removed, and `batch_size` can be set accordingly.
```python
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:0")

res = model.generate(
    input=input_file,
    cache={},
    language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    batch_size=64, 
)
```

For more usage, please ref to [docs](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md)



### Export and Test

```python
# pip3 install -U funasr-onnx
from funasr_onnx import SenseVoiceSmall

model_dir = "iic/SenseVoiceCTC"
model = SenseVoiceSmall(model_dir, batch_size=1, quantize=True)

wav_path = [f'~/.cache/modelscope/hub/{model_dir}/example/asr_example.wav']

result = model(wav_path)
print(result)
```

## Service

Undo

## Finetune

### Requirements

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
```

### Data prepare

Data examples

```text
{"key": "YOU0000008470_S0000238_punc_itn", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|withitn|>", "target": "Including legal due diligence, subscription agreement, negotiation.", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/YOU0000008470_S0000238.wav", "target_len": 7, "source_len": 140}
{"key": "AUD0000001556_S0007580", "text_language": "<|en|>", "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>", "target": "there is a tendency to identify the self or take interest in what one has got used to", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/industrial_data/english_all/audio/AUD0000001556_S0007580.wav", "target_len": 18, "source_len": 360}
```

Full ref to `data/train_example.jsonl`

### Finetune

Ensure to modify the train_tool in finetune.sh to the absolute path of `funasr/bin/train_ds.py` from the FunASR installation directory you have set up earlier.

```shell
bash finetune.sh
```

## WebUI

```shell
python webui.py
```

<div align="center"><img src="image/webui.png" width="700"/> </div>

<a name="Community"></a>
# Community


