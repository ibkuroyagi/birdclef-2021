PYTHON:= python3.6
CUDA_VERSION:= 11.1
PYTORCH_VERSION:= 1.8.1
DOT:= .
.PHONY: all clean

all: virtualenv

centos:
	virtualenv -p $(PYTHON) centos_venv
	. centos_venv/bin/activate; pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
	. centos_venv/bin/activate; cd ../; pip install -r requirements.txt
	# . centos_venv/bin/activate; mkdir ../input/; cd ../input/; kaggle competitions download \
	# 	-c birdclef-2021
	# cd ../input/; unzip birdclef-2021.zip -d birdclef-2021; \
	# 	rm birdclef-2021.zip
	# . centos_venv/bin/activate; mkdir ../input/; cd ../input/; kaggle datasets download \
	# 	-d yasufuminakama/iterative-stratification
	# cd ../input/; unzip iterative-stratification.zip; rm iterative-stratification.zip
virtualenv:
	./setup_anaconda.sh venv conda_env
	. ./activate_python.sh; conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
	. ./activate_python.sh; conda env update -n conda_env -f conda_env.yaml
	. ./activate_python.sh; mkdir ../input/; cd ../input/; kaggle competitions download \
		-c birdclef-2021
	cd ../input/; unzip birdclef-2021.zip -d birdclef-2021; \
		rm birdclef-2021.zip
apex:
	export CUDA_HOME=/usr/local/cuda-$(CUDA_VERSION)/
	git clone https://github.com/NVIDIA/apex.git
	. venv/bin/activate; cd apex; \
		pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

clean:
	rm -fr venv apex
	find -iname "*.pyc" -delete
