#!/bin/bash

xvfb-run -s "-screen 1 1400x900x24" jupyter notebook --ip=0.0.0.0 --allow-root
