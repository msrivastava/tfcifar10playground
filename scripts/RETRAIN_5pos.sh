export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_train \
--custom_overrides True \
--retrain True \
--retrain_list 'conv2 norm2 pool2 local3 local4' \
--data_url https://dl.dropbox.com/s/csavffd52et4l0l/RETRAIN_5pos_24x24.zip \
--data_file_root RETRAIN_5pos_24x24 \
--zip_datafile True \
--train_file train.bin \
--num_train_files 1 \
--eval_file test_4pos.bin \
--data_dir ../tfdata/mcifar10_5pos_data \
--checkpoint_dir ../tfdata/aws_cifar10_train \
--image_size 24 \
--input_image_height 24 \
--input_image_width 24 \
--label_bytes 1 \
--random_crop False \
--random_flip False \
--num_classes 101 \
--num_examples_per_epoch_for_train 6060 \
--num_examples_per_epoch_for_eval 1515 \
--batch_size 64 \
--initial_learning_rate 0.1 \
--learning_rate_decay_factor 0.1 \
--num_epochs_per_decay 350.0 \
--moving_average_decay 0.9999 \
--max_steps 20000 \
--debug \
--train_dir ../tfdata/mcifar10_5pos_layer1frozen_train

export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_eval \
--data_url https://dl.dropbox.com/s/csavffd52et4l0l/RETRAIN_5pos_24x24.zip \
--data_file_root RETRAIN_5pos_24x24 \
--data_dir ../tfdata/mcifar10_5pos_data \
--checkpoint_dir ../tfdata/mcifar10_5pos_layer1frozen_retrain \
--eval_data test \
--eval_file test_5pos.bin \
--custom_overrides True \
--zip_datafile True \
--train_file train.bin \
--num_train_files 1 \
--image_size 24 \
--input_image_height 24 \
--input_image_width 24 \
--label_bytes 1 \
--random_crop False \
--random_flip False \
--num_classes 101 \
--num_examples_per_epoch_for_train 6060 \
--num_examples_per_epoch_for_eval 1515 \
--num_examples 1515 \
--batch_size 64 \
--initial_learning_rate 0.1 \
--learning_rate_decay_factor 0.1 \
--num_epochs_per_decay 350.0 \
--moving_average_decay 0.9999 \
--max_steps 20000 \
--debug \
--eval_dir ../tfdata/mcifar10_5pos_eval

export PYTHONPATH=mcifar10 ; python -m mcifar10.cifar10_eval \
--data_url https://dl.dropbox.com/s/csavffd52et4l0l/RETRAIN_5pos_24x24.zip \
--data_file_root RETRAIN_5pos_24x24 \
--data_dir ../tfdata/mcifar10_5pos_data \
--checkpoint_dir ../tfdata/mcifar10_5pos_layer1frozen_retrain \
--eval_data test \
--eval_file test_4pos.bin \
--custom_overrides True \
--zip_datafile True \
--train_file train.bin \
--num_train_files 1 \
--image_size 24 \
--input_image_height 24 \
--input_image_width 24 \
--label_bytes 1 \
--random_crop False \
--random_flip False \
--num_classes 101 \
--num_examples_per_epoch_for_train 6060 \
--num_examples_per_epoch_for_eval 1515 \
--num_examples 1515 \
--batch_size 64 \
--initial_learning_rate 0.1 \
--learning_rate_decay_factor 0.1 \
--num_epochs_per_decay 350.0 \
--moving_average_decay 0.9999 \
--max_steps 20000 \
--debug \
--eval_dir ../tfdata/mcifar10_5pos_eval