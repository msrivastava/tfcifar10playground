# make a copy of cifar10 package that is installed
cp -p -r ~/anaconda/lib/python2.7/site-packages/tensorflow/models/image/cifar10 .

# rename the folder
mv cifar10 mcifar10

# remove all compiled files
rm mcifar10/*.pyc

# Now systematally change all occurrences of
# "from tensorflow.models.image.cifar10” with “from mifar10”
sed -i '' 's/from tensorflow.models.image.cifar10/from mcifar10/g'  mcifar10/*.py

# Finally test running it...
export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train 
