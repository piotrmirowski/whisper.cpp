# talk-server

Talk with a server, sending requests to a specified URL.

## Building

The `talk-server` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2 on Linux
sudo apt-get install libsdl2-dev

# Install SDL2 on Mac OS
brew install sdl2

# Build the "talk" executable
make talk-server
```

## Running the server


Print usage:
```bash
./talk-server -h
```

Assuming that:
** you have a server running at `http://localhost:8888/speech` that accepts `POST` requests with key, value pair `text`, `RESULT_OF_SPEECH_TRANSCRIPTION`,
** that you want to use the `tiny.en` binarised model for speech recognition,
** and that the microphone device is 1, then run:

```bash
./talk-server -c 1 -mw models/ggml-tiny.en-q5_0.bin -u http://localhost:8888/speech
```

The program recovers gracefully from failed server connections.
