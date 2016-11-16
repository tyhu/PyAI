export PYTHONPATH=/home/harry/github/coco-caption:/home/harry/github/Jobman$PYTHONPTH
export PATH=/home/harry/github/Jobman/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cudnn/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cudnn/include:$CPATH
export LIBRARY_PATH=/usr/local/cudnn/lib64:$LD_LIBRARY_PATH
export THEANO_FLAGS='cuda.root=/usr/local/cuda/,device=gpu0,floatX=float32'
python $1
