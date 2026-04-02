import cv2
import numpy as np
from skimage import restoration
from pathlib import Path
import argparse

def get_motion_psf(size=15, angle=0):
    """Creates a linear motion blur kernel."""
    psf = np.zeros((size, size))
    center = size // 2
    cv2.line(psf, (0, center), (size - 1, center), 1, thickness=1)
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    psf = cv2.warpAffine(psf, M, (size, size))
    return psf / psf.sum()

def deblur_richardson_lucy_rgb(image, psf, iterations=20):
    """Deblurs an RGB image with edge-padding to prevent ringing artifacts."""
    pad_size = psf.shape[0]
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')
    restored = np.zeros_like(padded_img, dtype=np.float64)
    
    for c in range(3):
        channel = padded_img[:, :, c] / 255.0
        restored[:, :, c] = restoration.richardson_lucy(channel, psf, num_iter=iterations)

    restored_cropped = restored[pad_size:-pad_size, pad_size:-pad_size, :]
    return np.clip(restored_cropped * 255, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restoration Script')
    parser.add_argument('--input', type=str, required=True, help='Path to blurred image')
    parser.add_argument('--output', type=str, required=True, help='Path to save restored image')
    parser.add_argument('--iter', type=int, default=20, help='Number of iterations')
    args = parser.parse_args()

    img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
    psf = get_motion_psf(size=15, angle=0)
    deblurred = deblur_richardson_lucy_rgb(img, psf, iterations=args.iter)
    
    cv2.imwrite(args.output, cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR))
    print(f'Successfully deblurred and saved to {args.output}')
