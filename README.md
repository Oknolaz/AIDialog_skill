# AIDialog skill for Vasisualy voice assistant
**AIDialog** is a russian language conversational skill for Vasisualy voice assistant using the pretrained [RudialoGPT3 AI model](https://github.com/Grossmend/DialoGPT).

:warning: **This skill uses a large amount of RAM, heavily loads the processor and slows down the launch of the Vasisualy.**
:warning: **It is recommended to have at least 6 Gb of RAM.**
## Supported platforms
- **GNU/Linux (Qt5, CLI)**
- **Windows** (Not tested)
## Installation
Clone this repo in `AIDialog` dir and move it to `vasisualy/skills` directory in the project dir:
```
git clone https://github.com/oknolaz/AIDialog_skill AIDialog
```
And then install requirements:
```
pip3 install torch transformers
```
## How to use
Just say to Vasisualy something like this: "Давай поговорим", "Поболтаем", "Побазарим"...
