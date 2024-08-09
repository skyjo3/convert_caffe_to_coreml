# convert_caffe_to_coreml

This is a simple Python script that converts a pre-trained Caffe model into a CoreML model.

There should be input files of:
1. A Caffe Model (e.g., example_oxford102.caffemodel)
2. A configuration file (e.g., example_deploy.prototxt)
3. A list of labels (e.g., example_flower_labels.txt)

The output would be in .mlmodel format.
