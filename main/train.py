# -*- coding: utf-8 -*-#
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import tensorflow as tf
import argparse
import model
from data_loader import loader
from util import env
from util import helper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    # parser.add_argument("--worker_hosts", help="Worker host list")
    parser.add_argument("--pooling", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--mode", type=int, help = "1:train 0:test")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--kid_classes", type=int, default=2)
    parser.add_argument("--hidden_size", type=str, default="50")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--query_length", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--word_dimension", type=int, default=100)
    parser.add_argument("--word_count", type=int, default=100)
    parser.add_argument("--snap_shot", type=int, default=10000)

    return parser.parse_known_args()[0]


def main():
    # Parse arguments and print them
    args = parse_args()
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Check if the model has already exisited
    model_save_dir = args.buckets + args.checkpoint_dir
    if tf.gfile.Exists(model_save_dir + "/checkpoint"):
        raise ValueError("Model %s has already existed, please delete them and retry" % model_save_dir)

    helper.dump_args(model_save_dir, args)

    convnet_model = model.SWEModel(
        model_configs=model.SWEModel.ModelConfigs(
            pooling = args.pooling,
            dropout=args.dropout,
            dim_word_embedding=args.word_dimension,
            init_with_w2v=False,
            hidden_layers=args.hidden_size,
            word_count=args.word_count,
            kid_count=args.kid_classes
        ),
        train_configs=model.TrainConfigs(
            learning_rate=args.learning_rate
        ),
        predict_configs=None,
        run_configs=model.RunConfigs(log_every=200)
    )

    estimator = tf.estimator.Estimator(
        model_fn=convnet_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=args.snap_shot,
            keep_checkpoint_max=100
        )
    )

    print("Start training......")
    estimator.train(
        loader.OdpsDataLoader(
            table_name=args.tables,
            max_length=args.query_length,
            mode=args.mode
        ).input_fn,
        steps=args.max_steps
    )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

