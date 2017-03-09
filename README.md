# How to use this?
## for TF 0.9
ln -s face_incr_cifar10_r0.9 mcifar10
## for TF 0.12
ln -s face_incr_cifar10_r0.12 mcifar10

## train
export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train
## retrain
export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train --retrain

## use command line options to specify new URL etc.
export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train --data_url DATA_URL --train_file TRAIN_FILE --eval_file EVAL_FILE --image_size IMAGE_SIZE -num_classes NUM_CLASSES --num_examples_per_epoch_for_train NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN --num_examples_per_epoch_for_eval NUM_EXAMPLES_PER_EPOCH_FOR_EVAL --num_batches NUM_BATCHES  --batch_size BATCH_SIZE

# How to add a new cifar10 package corresponding to a new release of TF?

## make a copy of cifar10 package that is installed
## e.g. if it is from Anaconda, then:
cp -p -r ~/anaconda/lib/python2.7/site-packages/tensorflow/models/image/cifar10 .

## rename the folder
mv cifar10 cifar10_rN.N
ln -s cifar10_rN.N mcifar10

## remove all compiled files
rm mcifar10/*.pyc

## Now systematically change all occurrences of "from tensorflow.models.image.cifar10” to “from mifar10”
sed -i '' 's/from tensorflow.models.image.cifar10/from mcifar10/g'  mcifar10/*.py

## Finally test running it...
export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train
