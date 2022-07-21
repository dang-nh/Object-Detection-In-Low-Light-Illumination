from mmdet.apis import init_detector, inference_detector
from models.image_enhancement.LDR2HDR.enhance import enhance
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='input image directory')
    parser.add_argument('--output_dir', type=str, help='output image directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_mAP_epoch_43.pth', help='checkpoint path')
    parser.add_argument('--cfg', type=str, default='/home/ubuntu/thanh.nt176874/dangnh/Object-Detection-In-Night-Vision/configs/yolov3/yolov3_only_image_process.py', help='config file path')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    enhance(args.input_dir, args.input_dir+'/enhanced')

    model = init_detector(args.cfg, args.checkpoint, device='cpu')
    
    for img in tqdm(os.listdir(args.input_dir + '/enhanced')):
        img_path = os.path.join(args.input_dir + '/enhanced', img)

        result = inference_detector(model, img_path)
        # or save the visualization results to image files
        model.show_result(img_path, result, out_file=os.path.join(args.output_dir, img))
