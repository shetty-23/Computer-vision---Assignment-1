from ultralytics import YOLO
import argparse

def run_eval(weights_path, data_yaml='../yolo_test_eval/eval_sharp.yaml'):
    model = YOLO(weights_path)
    print(f'Running official validation using {weights_path}...')
    metrics = model.val(data=data_yaml, project='Task5_Eval', name='final_report')
    print(f'mAP50: {metrics.box.map50:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='../yolo_test_eval/eval_sharp.yaml')
    args = parser.parse_args()
    run_eval(args.weights, args.data)
