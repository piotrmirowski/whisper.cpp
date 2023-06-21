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

# Run it
./talk-server
```
