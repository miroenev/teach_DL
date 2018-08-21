#!/bin/bash
pip install ipywidgets 
jupyter nbextension enable --py widgetsnbextension
pip install ipydatawidgets 
pip install k3d
jupyter nbextension install --py --sys-prefix k3d
jupyter nbextension enable --py --sys-prefix k3d