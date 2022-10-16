import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
image_path = r'/data/unagi0/cui_data/Japan-Project-cui/Oct12_down2'
save_path = r'/data/unagi0/cui_data/Japan-Project-cui/Oct12_detection_COCO'
mkdir(save_path)

def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    for file in os.listdir(image_path):
        whole_dir = os.path.join(image_path, file)
        save_dir = os.path.join(save_path, file)
        print(whole_dir)
        result = inference_detector(model, whole_dir)
        # show the results
        show_result_pyplot(model, whole_dir, result, save_dir=save_dir, score_thr=args.score_thr)


if __name__ == '__main__':
    main()

