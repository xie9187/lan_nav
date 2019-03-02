#!/bin/bash

nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
nohup Xorg :7 &
/opt/TurboVNC/bin/vncserver :8
