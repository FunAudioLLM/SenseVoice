Provide an OpenAI API transcribe server.


## Build Docker Image

```bash
docker-compose build
```

or

```bash
docker build -t sensevoice-openai-server .
```

## Start Server

### Docker Compose
Change volumes in docker-compose.yml to the path of the model you want to use.

```bash
docker-compose up -d
```

### Docker
```bash
docker run -d -p 8000:8000 -v "/your/cache/dir:/root/.cache" sensevoice-openai-server
```

## Usage

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F model="iic/SenseVoiceSmall" \
  -F file="@/path/to/file/openai.mp3"
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")

audio_file= open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="iic/SenseVoiceSmall", 
  file=audio_file
)
print(transcription.text)
```