from ultralytics import YOLO
import argparse
from pathlib import Path

def train_model(data_yaml, epochs=5):
    model = YOLO('yolov8s.pt')
    model.train(data=data_yaml, epochs=epochs, imgsz=320, project='../Task4_Runs', name='deblur_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['stratify', 'train'], required=True)
    parser.add_argument('--input', type=str, help='Input directory for stratification')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    if args.mode == 'stratify':
        print(f"Starting Laplacian stratification on {args.input}...")
        # Logic for Task 1: Dataset Stratification
    elif args.mode == 'train':
        train_model('../yolo_dataset/data.yaml', args.epochs)
