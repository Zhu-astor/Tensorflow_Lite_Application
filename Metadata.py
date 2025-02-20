from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os

# 創建模型資訊
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "Image Reflection Removal"
model_meta.description = ("_outputisfeature A model that removes reflections from images using a Pix2Pix architecture with Sobel filter for attention.")
model_meta.version = "v4"
model_meta.author = "Zhuastor"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

#input type:image 
#output type:feature

# 創建輸入 Tensor 資訊
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to the model. The expected image is {0} x {1}, with "
    "three channels (red, green, and blue) per pixel. The input values should be normalized between -1 and 1.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

# 設定輸入的 normalization 處理
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta.processUnits = [input_normalization]

# 設定輸入的統計數據
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [-255]
input_meta.stats = input_stats

# # 創建輸出 Tensor 資訊
# output_meta = _metadata_fb.TensorMetadataT()
# output_meta.name = "output_image"
# output_meta.description = "The output is an image with reflections removed, normalized between -1 and 1."
# output_meta.content = _metadata_fb.ContentT()
# output_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
# output_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
# output_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

# # 設定輸出的統計數據
# output_stats = _metadata_fb.StatsT()
# output_stats.max = [255]
# output_stats.min = [-255]
# output_meta.stats = output_stats

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 1001 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [255.0]
output_stats.min = [-255.0]
output_meta.stats = output_stats
# 創建 Subgraph 資訊
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

# 生成 metadata
b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

# 填充 metadata 到 TFLite 模型中
model_file = "C:/Users/bubbl/Desktop/Contest/AI GO/GAN_Test/saved_model_14_at_d2/generator_500_3.tflite"
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.populate()

# 保存模型 metadata 到檔案中
metadata_file = model_file + ".metadata"
with open(metadata_file, 'wb') as f:
    f.write(metadata_buf)

print("Metadata successfully populated and saved.")
