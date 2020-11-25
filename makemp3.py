# -*- coding: utf-8 -*-
from gtts import gTTS



tts = gTTS(text="get out of the way! ", lang='en')
tts.save("notice0.mp3")