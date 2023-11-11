import logging
from run.Inference import _args, load_config, inference

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    args = _args()
    args.train_months = 8
    args.eval_months = 3
    opt = load_config(args.config)

    result = inference(opt, args, logger)
