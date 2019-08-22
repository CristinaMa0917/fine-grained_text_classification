# -*- coding: utf-8 -*-#
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import time
import tensorflow as tf
import argparse
import model
from data_loader import loader
from util import env
from util import helper
from data_dumper import dumper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--outputs", type=str, help="Destination of the table ")
    parser.add_argument("--buckets", type=str, help="Worker task index")
    # parser.add_argument("--worker_hosts", help="Worker host list")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--output_embedding", type=int, default=0, help="Indicating whether outputting embedding")
    parser.add_argument("--output_score", type=int, default=0, help="Indicating whether outputting scores")
    parser.add_argument("--output_prediction", type=int, default=1, help="Indicating whether outputting prediction")
    parser.add_argument("--mode", type=int, help="1:train 0:inference")
    parser.add_argument("--query_length", type=int, default=1200)
    # Arguments for distributed inference
    parser.add_argument("--task_index", type=int, help="Task index")
    parser.add_argument("--ps_hosts", type=str, help="")
    parser.add_argument("--worker_hosts", type=str, help="")
    parser.add_argument("--job_name", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--step", type=int)

    return parser.parse_known_args()[0]


def _do_prediction(result_iter, writer, args, model_args):
    def _oneid_extractor(prediction, record):
        record.append(str(prediction["oneid"]))

    def _feature_extractor(prediction, record):
        record.append(prediction["feature"])

    def _confidence_extractor(prediction, record):
        record.extend(prediction['confidence'])

    def _prediction_extractor(prediction, record):
        record.append(prediction["prediction"])

    print("Start inference......")
    t_start = t_batch_start = time.time()
    report_gap = 10000

    transformers = []
    n_fields = 1
    transformers.append(_oneid_extractor)

    if args.output_embedding:
        transformers.append(_feature_extractor)
        n_fields += 1

    if args.output_prediction:
        transformers.append(_prediction_extractor)
        n_fields += 1

    if args.output_score:
        transformers.append(_confidence_extractor)
        n_fields += model_args.classes
    indices = [i for i in range(n_fields)]

    for i, prediction in enumerate(result_iter, 1):
        record = []
        for extractor in transformers:
            extractor(prediction, record)

        writer.write(record, indices)
        if i % report_gap == 0:
            t_now = time.time()
            print("[{}]Processed {} samples, {} records/s, cost {} s totally, {} records/s averagely".format(
                args.task_index,
                i,
                report_gap / (t_now - t_batch_start),
                (t_now - t_start),
                i / (t_now - t_start)
            ))
            t_batch_start = t_now

    writer.close()



def main():
    args = parse_args()
    print("Main arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))

    # Setup distributed inference
    dist_params = {
        "task_index": args.task_index,
        "ps_hosts": args.ps_hosts,
        "worker_hosts": args.worker_hosts,
        "job_name": args.job_name
    }
    slice_count, slice_id = env.set_dist_env(dist_params)

    # Load model arguments
    model_save_dir = args.buckets + args.checkpoint_dir
    model_args = helper.load_args(model_save_dir)

    convnet_model = model.SWEModel(
        model_configs=model.SWEModel.ModelConfigs(
            pooling=model_args.pooling,
            dropout=0,
            dim_word_embedding=model_args.word_dimension,
            init_with_w2v=False,
            hidden_layers=model_args.hidden_size,
            classes=model_args.classes,
            word_count=model_args.word_count,

        ),
        train_configs=None,
        predict_configs=model.PredictConfigs(
            output_confidence=bool(args.output_score),
            output_embedding=bool(args.output_embedding),
            output_prediction=bool(args.output_prediction)
        ),
        run_configs=None
    )

    checkpoint_path = None
    if args.step > 0:
        checkpoint_path = model_save_dir + "/model.ckpt-{}".format(args.step)

    estimator = tf.estimator.Estimator(
        model_fn=convnet_model.model_fn,
        model_dir=model_save_dir,
        config=tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
            save_checkpoints_steps=model_args.max_steps,
            keep_checkpoint_max=1
        )
    )

    result_iter = estimator.predict(
        loader.OdpsDataLoader(
            table_name=args.tables,
            max_length=args.max_length,
            mode = 0,
            slice_id=slice_id,
            slice_count=slice_count,
            shuffle=0,
            repeat=1,
            batch_size=64
        ).input_fn,
        checkpoint_path=checkpoint_path
    )

    odps_writer = dumper.get_odps_writer(
        args.outputs,
        slice_id=slice_id
    )
    _do_prediction(result_iter, odps_writer, args, model_args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
