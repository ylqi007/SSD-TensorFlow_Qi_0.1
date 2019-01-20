# SSD-TensorFlow_Qi_0.1

## 1. Generate TFRecord files

`tf_convert_data.py`                        <br />
    |--> `pascalvoc_to_tfrecords.run()`     <br />
        |--> `_add_to_tfrecord`             <br />
                |--> `_process_image`       <br />
                |--> `_convert_to_example`

## 2. Get data from TFRecord files
dataset = `dataset_factory.get()`                           <br />
            |--> `pascalvoc_2007.get_split()`               <br />
                    |--> `pascalvoc_common.get_split()`     <br />
                            |--> return `slim.dataset.Dataset`  <br />

