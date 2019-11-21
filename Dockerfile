FROM nvcr.io/nvidia/tensorflow:19.10-py3

RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install ipydatawidgets 
RUN pip install k3d
RUN jupyter nbextension install --py --sys-prefix k3d
RUN jupyter nbextension enable --py --sys-prefix k3d

