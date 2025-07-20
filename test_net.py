import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference,training_neat_eval
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFRNet Testing")
    parser.add_argument(
        "--config_file", default="/data/Code/MFRNet-master/configs/MSVR310/MFRNet.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--model_path", default="/data/object/Out_MFRNet/MSVR310_MFRNetbest.pth", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("MFRNet", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.eval()
    model.load_param(args.model_path)

    # from thop import profile
    # import torch
    # number=1
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # data = torch.randn([number, 3, 256, 128]).to(device)
    # img = {'RGB': data,'NI': data,'TI': data}
    # camids = torch.ones([number, ]).to(torch.int64).to(device)
    # target_view = torch.ones([number, ]).to(torch.int64).to(device)
    # return_pattern = 3
    # flops, params = profile(model, inputs=(img, None, camids, -1*target_view, return_pattern,))
    # print(flops / 1e9, params / 1e6)

    do_inference(cfg, model, val_loader, num_query, return_pattern=3)
