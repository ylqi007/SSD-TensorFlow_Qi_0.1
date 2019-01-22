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

## 3. Get anchors from `SSDNet.anchors()`
`ssd_anchors = ssd_net.anchors(ssd_shape)`      <br />
    |--> `ssd_anchors_all_layers()`             <br />
            |--> `ssd_anchor_one_layer()`       <br />

## 4. Get image and annotation info from provider of dataset.
```
# Get for SSD network: image, labels, bboxes.
[image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                 'object/label',
                                                 'object/bbox'])
```

## 5. image_preprocess_fn()


## 6. ssd_net.bboxes_encode()
```
gclasses, glocalisations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
```

## 7. ssd_net.losses()
```
ssd_net.losses(logits, localisations,
               b_gclasses, b_glocalisations, b_gscores,
               match_threshold=FLAGS.match_threshold,
               negative_ratio=FLAGS.negative_ratio,
               alpha=FLAGS.loss_alpha,
               label_smoothing=FLAGS.label_smoothing)
```
