import argparse

def option():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=int, default="2")

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to save logs')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model name')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', type=str, default='', help='path to checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', help='freeze encoder weights')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--pretrained_path', type=str, default='', help='path to pretrained model')
    parser.add_argument('--num_frozen_layers', type=int, default=0, help='number of frozen layers')
    parser.add_argument('--num_freeze_layers', type=int, default=0, help='number of freeze layers')
    parser.add_argument('--num_freeze_layers_encoder', type=int, default=0, help='number of freeze layers')
    
    return parser.parse_args()