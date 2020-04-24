To do list:
***Add SSH key to not have to type in password???
***make log file output

git commands:
git status
git add .
git commit -m "comment"
git push
git pull

rm -f .git/index.lock ==> if crashes during push/pull need to remove this file manually
git config --global user.email "tigerxu96@hotmail.com"
git config --global user.name "yxu233"

Compute canada:
module load python/3.6
virtualenv --no-download tensorflow
source tensorflow/bin/activate
pip install --no-index tensorflow_gpu==1.15
###pip install --no-index tensorflow_gpu

Job submission:
#!/bin/bash
#SBATCH --gres=gpu:1   # requests GPU "generic resource"
#SBATCH --cpus-per-task=10 # max CPU cores per GPU request (6 on Cedar, 16 on Graham).
#SBATCH --mem=80000M   # memory per node (in mbs)
#SBATCH --time=02-00:00 # time (DD-HH:MM) OR (hh:mm:ss) format
#SBATCH --output=%N-%j.out # %N for node name, %j for jobID

module load cuda cudnn python/3.5   # load python and then tensorflow
source tensorflow/bin/activate
python ./CARE.py --variable_update=parameter_server --local_parameter_device=cpu # script to run (change to Myelin UNet)


Installation of UNet:
pip install matplotlib scipy scikit-image pillow numpy natsort opencv-python keras pandas bcolz skan Cython sklearn numba

pip install numba

***installing sklearn requires Cython!

(SKAN ==> requires pandas, numba... ==> DOES NOT WORK ON BELUGA, numba is broken)

# *** pip install tifffile NO LONGER WORKS

mahotas? - failed ==> ***April 2002, now works with pip install mahotas, but do we need it?
conda config --add channels conda-forge
conda install mahotas

Graphics card driver
CUDA Toolkit ==> needs VIsual studio (base package sufficient???)
CuDnn SDK ==> 

Ignore step about putting cudnn with Visual Studio

For CARE model:
pip install numexpr ==> for normalization with csbdeep
need tensorflow version >= 1.14 to have correct tf.msssim function... "filter_size" parameter

New imports:
pip install bcolz
pip install skan *** NEW!!! allows skeleton analysis


Things to remove:
- tifffile
