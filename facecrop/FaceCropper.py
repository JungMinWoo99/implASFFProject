import os

from face_crop_plus import Cropper

cropper = Cropper(output_size=256, face_factor=0.67, device="cuda")

if __name__ == '__main__':
    index = 0
    start_dir = "D:/system/Dataset/VGGFace2"
    for root, dirs, files in os.walk(start_dir):
        for dir in dirs:
            input_dir_path = os.path.join(root, dir)
            output_dir_path = os.path.join("E:/code_depository/depository_python/FSR_project/ImpASFF/facecrop/VGGFace2_crop", str(index))
            os.makedirs(output_dir_path,exist_ok=True)
            index += 1

            cropper.process_dir(input_dir=input_dir_path, output_dir=output_dir_path)
