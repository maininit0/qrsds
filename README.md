# 人体关键点识别

# humble

source /opt/tros/humble/setup.bash

\# 打开摄像头
export CAM\_TYPE=usb

# 启动关键点检测

ros2 launch mono2d\_body\_detection mono2d\_body\_detection.launch.py

# 跟随移动

cd /userdata/pyx\_ws

source /opt/tros/humble/setup.bash
python3 huagui.py

# 模仿动作

cd /userdata/pyx\_ws

source /opt/tros/humble/setup.bash
python3 body.py

# 动作捕获

cd /userdata/pyx\_ws

source /opt/tros/humble/setup.bash
python3 shipin.py



\# 动作复刻
cd /userdata/pyx\_ws

source /opt/tros/humble/setup.bash
python3 fuke.py
