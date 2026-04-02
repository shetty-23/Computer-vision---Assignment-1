import cv2
from ultralytics import YOLO
import argparse

def run_baseline_analysis(image_path, model_path='yolov8n.pt'):
    model = YOLO(model_path)
    img = cv2.imread(str(image_path))
    results = model.predict(img, conf=0.25)
    print(f'Detected {len(results[0].boxes)} objects in {image_path}')
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    args = parser.parse_args()
    run_baseline_analysis(args.input, args.weights)
