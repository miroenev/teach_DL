FROM nvcr.io/nvidia/mxnet:18.07-py3

RUN apt-get update && apt-get install -qqy x11-apps \
	python-numpy python-dev cmake \
	zlib1g-dev libjpeg-dev xvfb ffmpeg \
	xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
	python3 get-pip.py --force-reinstall 
#	python get-pip.py --force-reinstall

RUN git clone https://github.com/openai/gym && cd gym && \
	python3 ./setup.py clean && \
	pip3 install -e '.[all]'

RUN pip3 install keras-mxnet \
	jupyter \
	matplotlib \
	ipywidgets \
	ipydatawidgets \
	k3d

ENV KERAS_BACKEND=mxnet

RUN jupyter nbextension enable --py widgetsnbextension && \
	jupyter nbextension install --py --sys-prefix k3d && \
	jupyter nbextension enable --py --sys-prefix k3d
	

# sudo nvidia-docker run -p 8888:8888 mxez:0 jupyter notebook --allow-root --no-browser --ip 0.0.0.0

# xvfb-run -s "-screen 800x600x24" jupyter notebook
