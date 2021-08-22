python3 mo.py --input_model /home/retina/retinaface-res50-gender-320.onnx --output_dir /home/retina/res50  --data_type FP16  --input_shape [1,3,320,320] --mean_values [104,117,123] --scale_values [57,57,58] --reverse_input_channels --move_to_preprocess

python3 mo.py --input_model /home/retina/retinaface-mbv2-320.onnx --output_dir /home/retina/mbv2  --data_type FP16  --input_shape [1,3,320,320] --mean_values [104,117,123] --scale_values [57,57,58] --reverse_input_channels --move_to_preprocess
