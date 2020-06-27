# tensorflow
How to install Tensorflow 1.15 GPU from SOURCE with CUDA 11, cuDNN 8.0.1 with most recent nvidia driver for python on Ubuntu 18.04 LTS
### This is going to be a tutorial on how to install tensorflow 1.15 GPU version. We will also be installing CUDA 11 and cuDNN 8.0.1 along with tensorflow 1.15.
### In order to use the GPU version of TensorFlow, you will need an NVIDIA GPU with a compute capability > 3.0. Check the GPU consistency with the latest nvidia driver as well.

# Step 1: Update and Upgrade your system:

	sudo apt-get update 
	sudo apt-get upgrade

# Step 2: Verify You Have a CUDA-Capable GPU:

	lspci | grep -i nvidia

# Step 3: Verify You Have a Supported Version of Linux:

	uname -m && cat /etc/*release

## The x86_64 line is a must have 64-bit linux

# Step 4: Install Dependencies:

	sudo apt-get install build-essential 
	sudo apt-get install cmake git unzip zip
	sudo apt-get install python-dev python3-dev python-pip python3-pip

## These are essentials for building from source.

# Step 5: Install linux kernel header:

	sudo apt-get install linux-headers-$(uname -r)

# Step 6: Install NVIDIA CUDA 11:

	sudo apt install build-essential gcc-6 g++-6
	sudo update-alternatives --remove-all gcc
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 20
	sudo update-alternatives --set gcc /usr/bin/gcc-6
	sudo update-alternatives --remove-all g++
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 20
	sudo update-alternatives --set g++ /usr/bin/g++-6

	sudo apt-get purge nvidia*
	sudo apt-get autoremove
	sudo apt-get autoclean
	sudo rm -rf /usr/local/cuda*

	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
	echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/ap/sources.list.d/cuda.list
	sudo apt-get update 
	sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-0 cuda-drivers
## or
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
	sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
	sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
	sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
	sudo apt-get update
	sudo apt-get -y install cuda

	sudo modprobe -r nouveau
	sudo modprobe -i nvidia

## set system wide paths
	echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda/bin"' | sudo tee /etc/environment
	echo /usr/local/cuda-11.0/lib64 | sudo tee /etc/ld.so.conf.d/cuda-11.0.conf
	sudo ldconfig

# Step 7: Reboot the system to load the NVIDIA drivers.

	sudo reboot

# Step 8: Go to terminal and type:

	echo 'export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
	echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
	source ~/.bashrc
	sudo ldconfig

## test your graphics OKAY for nvidia FAIL for nouveau:
	lsmod | grep nouv && echo FAIL || echo OKAY
	lsmod | grep nvid && echo OKAY || echo FAIL
	grep -E 'NVIDIA.*450.[0-9]+' /proc/driver/nvidia/version &>/dev/null && echo OKAY || echo FAIL
	nvcc -V | grep -E "V11.0.[0-9]+" &>/dev/null && echo OKAY || echo FAIL

## this should return stats for all installed cards
    nvidia-smi
## If you got nvidia-smi is not found (improbable) then you have unsupported linux kernel installed: 
	echo $(uname -r) 

## You can check your cuda installation using following sample:

	cuda-install-samples-11.0.sh ~
	cd ~/NVIDIA_CUDA-11.0_Samples/5_Simulations/nbody
	make
	./nbody

## Quite COOL right!!! try the other simulations also, it makes you appreciate your NVIDIA GPU if you're not into Vgames.
## Now let's get down to business...

# Step 9: Install cuDNN 8.0.1:


 ### Go to: NVIDIA cuDNN home page. https://developer.nvidia.com/cudnn
 ### Click Download. get cudnn-10.2-linux-x64-v7.6.5.32.tgz
 ### Complete the short survey and click Submit.
 ### Accept the Terms and Conditions. A list of available download versions of cuDNN displays.
 ### Select the cuDNN version you want to install. A list of available resources displays.

## Go to downloaded folder and in terminal perform following:
	tar -xzvf cudnn-11.0-linux-x64-v8.0.1.13.tgz
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

### (Next step is not essential since it can be defaulted in the Build later to NCCL1.3) 
### Do the same for NCCL: NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs. Get nccl_###+cuda10.2_x86_64.txz
	tar -xf nccl_###-#+cuda10.2_x86_64.txz
	cd nccl_###-#+cuda10.2_x86_64
	sudo cp -R * /usr/local/cuda-10.2/targets/x86_64-linux/
	sudo ldconfig

# Step 10: Install Dependencies:

## w/o virtual env:
	pip install keras==2.2.4
	pip3 install keras==2.2.4
	pip install -U --user pip six numpy wheel mock
	pip3 install -U --user pip six numpy wheel mock
	pip install -U --user keras_applications --no-deps
	pip3 install -U --user keras_applications --no-deps
	pip install -U --user keras_preprocessing --no-deps
	pip3 install -U --user keras_preprocessing --no-deps

## If in an active venv 'pip' is enough.

# Step 11: Tweaks:

## Do these tweaks to avoid getting errors in the bazel Build: {{don't take these for granted...and follow the pattern for other errors that might appear later due to missing library files!!!}}
	cd 
	cd  /usr/local/cuda-10.2/lib64
	sudo ln -s -T /usr/lib/x86_64-linux-gnu/libcublas.so.10.2.2.89 libcublas.so.10.2
	sudo ln -s -T libcusolver.so.10.3.0.89 libcusolver.so.10.2
	sudo ln -s -T libcurand.so.10.1.2.89 libcurand.so.10.2
	sudo ln -s -T libcufft.so.10.1.2.89 libcufft.so.10.2

## In case you decided not to install NCCL do:
	cd ../
	cd targets/x86_64-linux/include/
	sudo cp  /usr/include/cublas_v2.h .
## same for any similar c++ file missing. 


# Step 12: Configure Tensorflow from source:

## Download bazel:
	cd ~/
	wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh
	chmod +x bazel-0.24.1-installer-linux-x86_64.sh
	./bazel-0.24.1-installer-linux-x86_64.sh --user
	echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc

## Reload environment variables
	source ~/.bashrc
	sudo ldconfig

## Start the process of building TensorFlow by downloading latest tensorflow 1.15:
	cd ~/
	git clone https://github.com/tensorflow/tensorflow.git
	cd tensorflow
	git checkout r1.15

## One last TWEAK!!! 
## remove the line 
	"--bin2c-path=%s" % bin2c.dirname, 
## from the file third_party/nccl/build_defs.bzl.tpl

## Now:
	./configure

## Give python path in

Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

## Press enter two times

Do you wish to build TensorFlow with Apache Ignite support? [Y/n]: Y 

## (This 1st line might not appear)

Do you wish to build TensorFlow with XLA JIT support? [Y/n]: Y

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N

Do you wish to build TensorFlow with ROCm support? [y/N]: N

Do you wish to build TensorFlow with CUDA support? [y/N]: Y

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10.2

Please specify the location where CUDA 10.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 7.6.5

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda

Do you wish to build TensorFlow with TensorRT support? [y/N]: N

Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 1.3 
## (if you are lazy like me and didn't install a version of NCCL)

## Now we need compute capability which we have noted at step 1 eg. 5.0 check Nvidia website for GPU compute capabilities.

Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.1] 6.1
## e.g. GeForce GTX

Do you want to use clang as CUDA compiler? [y/N]: N

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc

Do you wish to build TensorFlow with MPI support? [y/N]: N

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native  .....something else]: -march=native
## (only -march=native)

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:N

## VOILA!!!  Configuration finished, now to the best part...


# Step 13: Build Tensorflow using bazel:

## To build a pip package for TensorFlow you would typically invoke the following command:
	bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

## Note:-

 add "--config=mkl" if you want Intel MKL support for newer intel cpu for faster training on cpu

 add "--config=monolithic" if you want static monolithic build (try this if build failed)

 add "--local_resources 2048,.5,1.0" if your PC has low ram causing Segmentation fault or other related errors
 
### This process will take a lot of time. It may take 3- 4 hours or maybe even more.
### Also if you got error like Segmentation Fault then try again it usually works. 

### The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the tensorflow_pkg directory:

	bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg

## Finally:

	cd tensorflow_pkg
	pip install tensorflow*.whl

## for python 2: (use sudo if required)

	pip2 install tensorflow*.whl

## for python 3: (use sudo if required)

	pip3 install tensorflow*.whl


# Step 14: Verify Tensorflow installation:

## Run in terminal

	python

## Now test if gpu works on an editor:

	import tensorflow as tf
	print('tfversion=',tf.__version__)

	with tf.device('/gpu:0'):
    	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    	c = tf.matmul(a, b)

	with tf.Session() as sess:
    	print(sess.run(c))


## All works? Congratulations!  You have now successfully installed tensorflow 1.15 GPU on your machine. 

## References
	https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/
	https://docs.nvidia.com/deeplearning/sdk/
	https://gist.github.com/DaneGardner/accd6fd330348543167719002a661bd5


Cheers!!
