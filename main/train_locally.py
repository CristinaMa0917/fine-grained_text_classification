# -*- coding: utf-8 -*-#
import tensorflow as tf
import argparse
import model
from data_loader import loader

tf.enable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Input table GUID")
    #parser.add_argument("--task_index", help="Worker task index")
    # parser.add_argument("--worker_hosts", help="Worker host list")
    parser.add_argument("--kernels", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Snapshot gap")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--hist_length", type=int, default=1200)
    parser.add_argument("--target_length", type=int, default=800)
    parser.add_argument("--hidden_size", type=str, default="50")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)

    return parser.parse_known_args()[0]


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the model has already exisited
    model_save_dir = args.checkpoint_dir
    if tf.gfile.Exists(model_save_dir + "/checkpoint"):
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)

    convnet_model = model.TextConvNet(
        model_configs=model.TextConvNet.ModelConfigs(
            kernels=args.kernels,
            dropout=args.dropout,
            dim_word_embedding=100,
            init_with_w2v=False,
            hidden_layers=args.hidden_size,
            word_count=178422
        ),
        train_configs=model.TrainConfigs(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        ),
        predict_configs=None,
        run_configs=model.RunConfigs(
            log_every=200
        )
    )

    estimator = tf.estimator.Estimator(
        model_fn=convnet_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=args.snapshot,
            keep_checkpoint_max=20
        )
    )

    print("Start training......")
    estimator.train(
        loader.LocalFileDataLoader(
            file_path=args.file_path,
            mode=tf.estimator.ModeKeys.TRAIN,
            hist_length=args.hist_length,
            target_length=args.target_length
        ).input_fn,
        steps=args.max_steps
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

