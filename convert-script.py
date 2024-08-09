# convert-script.py

import coremltools as ct

caffe_model = ('oxford102.caffemodel', 'deploy.prototxt')

labels = 'flower-labels.txt'

coreml_model = ct.converters.caffe.convert(
	caffe_model,
	class_labels=labels,
	image_input_names='data'
)

coreml_model.save('FlowerClassifier.mlmodel')