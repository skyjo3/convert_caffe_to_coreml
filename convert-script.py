# convert-script.py

import coremltools as ct
import os

input_dir = 'input'
output_dir = 'output'

caffe_model = (
    os.path.join(input_dir, 'example_oxford102.caffemodel'),
    os.path.join(input_dir, 'example_deploy.prototxt')
)
labels = os.path.join(input_dir, 'example_flower_labels.txt')

# Convert the Caffe model to a CoreML model
coreml_model = ct.converters.caffe.convert(
    caffe_model,
    class_labels=labels,
    image_input_names='data'
)

output_path = os.path.join(output_dir, 'example_FlowerClassifier.mlmodel')
coreml_model.save(output_path)

print(f"Model saved to {output_path}")