#!/bin/bash

XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -
sudo chmod 777 $XAUTH

#DISPLAY=172.17.0.1:10.0 
DISPLAY=`echo $DISPLAY | sed 's/^[^:]*\(.*\)/172.17.0.1\1/'`

X11PORT=`echo $DISPLAY | sed 's/^[^:]*:\([^\.]\+\).*/\1/'`
TCPPORT=`expr 6000 + $X11PORT`
sudo ufw allow from 172.17.0.0/16 to any port $TCPPORT proto tcp

sudo nvidia-docker run -ti --rm -e DISPLAY=$DISPLAY \
	-v $XAUTH:$XAUTH \
	-e XAUTHORITY=$XAUTH \
	-p 8888:8888 \
	gymp3x11jup:v0 bash	
# gym:py3 bash
# xvfb-run -s "-screen 0 800x600x24" jupyter notebook --allow-root --no-browser --ip 0.0.0.0

