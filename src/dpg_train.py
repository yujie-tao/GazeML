#!/usr/bin/env python3
"""Main script for training the DPG model for within-MPIIGaze evaluations."""
import argparse
import coloredlogs
import tensorflow as tf
from models import DPG
from datasources import HDF5Source

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(
        description='Train the Deep Pictorial Gaze model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    for i in range(0, 1):
        # Specify which people to train on, and which to test on
        # person_id = 'p%02d' % i
        # other_person_ids = ['p%02d' % j for j in range(15) if i != j]

        # Initialize Tensorflow session
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

            # Declare some parameters
            batch_size = 32

            model = DPG(
                session,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {
                            'combined_loss': ['hourglass', 'densenet'],
                        },
                        'metrics': ['gaze_mse', 'gaze_ang'],
                        'learning_rate': 0.0002,
                    },
                ],
                extra_tags=[],

                # Data sources for training (and testing).
                train_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=batch_size,
                        keys_to_use=['train'],
                        hdf_path='../datasets/world_cropped_contrast.hdf5',
                        eye_image_shape=(436, 504),
                        testing=False,
                        min_after_dequeue=30000,
                        staging=True,
                        shuffle=True,
                    ),
                },
                test_data={
                    'mpi': HDF5Source(
                        session,
                        data_format='NCHW',
                        batch_size=batch_size,
                        keys_to_use=['test'],
                        hdf_path='../datasets/world_cropped_contrast.hdf5',
                        eye_image_shape=(436, 504),
                        testing=True,
                    ),
                },
            )

            # Train this model for a set number of epochs
            model.train(
                num_epochs=20,
            )

            model.__del__()
            session.close()
            del session
