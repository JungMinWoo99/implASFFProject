import cv2
import numpy as np
import random
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma):
    """가우시안 커널을 생성하는 함수."""
    k = cv2.getGaussianKernel(size, sigma)
    gaussian = k @ k.T
    return gaussian


def apply_motion_blur(image, kernel):
    """이미지에 모션 블러를 적용하는 함수."""
    if len(image.shape) == 2:  # 그레이스케일 이미지
        return convolve2d(image, kernel, mode='same', boundary='wrap')
    elif len(image.shape) == 3:  # 컬러 이미지
        channels = [convolve2d(image[:, :, i], kernel, mode='same', boundary='wrap') for i in range(3)]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError('Unsupported image shape.')


def add_gaussian_noise(image, sigma):
    """이미지에 가우시안 노이즈를 추가하는 함수."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def jpeg_compression(image, quality):
    """이미지에 JPEG 압축을 적용하는 함수."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encoded_image, 1)
    return compressed_image


def downscale_image(image, scale_factor):
    """이미지를 다운스케일링하는 함수."""
    height, width = image.shape[:2]
    new_dimensions = (int(width / scale_factor), int(height / scale_factor))
    if new_dimensions[0] <= 0 or new_dimensions[1] <= 0:
        raise ValueError(f"Invalid scale factor {scale_factor} resulting in non-positive dimensions {new_dimensions}.")
    print(f"Downscaling from {image.shape[:2]} to {new_dimensions} with scale factor {scale_factor}")
    downscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return downscaled_image


def degradation_model(image_path,
                      # 블러 관련 파라미터
                      gaussian_blur_sigma_values=np.arange(1, 3.1, 0.1),
                      motion_blur_kernels=None,
                      # 다운스케일링 파라미터
                      scale_factor_values=np.arange(1, 8.1, 0.1),
                      # 노이즈 관련 파라미터
                      noise_sigma_values=range(0, 16),
                      # JPEG 압축 파라미터
                      jpeg_quality_values=range(10, 61)):
    """이미지에 다양한 열화 과정을 적용하는 함수."""

    # 이미지 로드
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Please check the path and file integrity.")

    # 이미지 색상 변환 (BGR -> RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 블러 타입 선택 (가우시안 블러 또는 모션 블러)
    if random.random() > 0.5:
        # 가우시안 블러 적용
        gaussian_blur_sigma = random.choice(gaussian_blur_sigma_values)
        kernel_size = int(2 * round(3 * gaussian_blur_sigma) + 1)
        kernel = gaussian_kernel(kernel_size, gaussian_blur_sigma)
        blurred_image = cv2.filter2D(image, -1, kernel)
        print(f"Applied Gaussian blur with sigma {gaussian_blur_sigma}")
    else:
        # 모션 블러 적용
        kernel = random.choice(motion_blur_kernels)
        blurred_image = apply_motion_blur(image, kernel)
        print("Applied motion blur")

    # 이미지 다운스케일링
    scale_factor = random.choice(scale_factor_values)
    downscaled_image = downscale_image(blurred_image, scale_factor)

    # 가우시안 노이즈 추가
    noise_sigma = random.choice(noise_sigma_values)
    noisy_image = add_gaussian_noise(downscaled_image, noise_sigma)
    print(f"Added Gaussian noise with sigma {noise_sigma}")

    # JPEG 압축 적용
    jpeg_quality = random.choice(jpeg_quality_values)
    degraded_image = jpeg_compression(noisy_image, jpeg_quality)
    print(f"Applied JPEG compression with quality {jpeg_quality}")

    return degraded_image


# 예시 모션 블러 커널 (더 추가하거나 파일에서 로드 가능)
motion_blur_kernels = [
    # 32개의 모션 블러 커널을 여기에 추가합니다
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # 예시 커널
    np.fliplr(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # 예시 커널 반전
    # 나머지 30개의 모션 블러 커널을 추가
]

if __name__ == '__main__':
    # 사용 예시
    image_path = "9.png"
    try:
        degraded_image = degradation_model(image_path, motion_blur_kernels=motion_blur_kernels)

        # 저품질 이미지 저장
        cv2.imwrite("low_quality_image.jpg", cv2.cvtColor(degraded_image, cv2.COLOR_RGB2BGR))
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)