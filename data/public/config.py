import numpy

from onnx_tool import create_ndarray_f32, create_ndarray_int64

public_models = {
    'folder': 'data/public',
    'models': [
        {
            'name': 'vae_encoder.onnx',
            'dynamic_input': None
        },
        {
            'name': 'vae_decoder.onnx',
            'dynamic_input': None
        },
        {
            'name': 'text_encoder.onnx',
            'dynamic_input': None
        },
        {
            'name': 'bertsquad-12.onnx',
            'dynamic_input':
                {
                    'unique_ids_raw_output___9:0': numpy.array((1,), dtype=numpy.int64),
                    'segment_ids:0': numpy.zeros((1, 256), dtype=numpy.int64),
                    'input_mask:0': numpy.zeros((1, 256), dtype=numpy.int64),
                    'input_ids:0': numpy.zeros((1, 256), dtype=numpy.int64),
                }
        },
        {
            'name': 'bvlcalexnet-12.onnx',
            'dynamic_input': None,
        },
        {
            'name': 'convnext_large.onnx',
            'dynamic_input':
                {
                    'input.1': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'efficientnet-lite4-11.onnx',
            'dynamic_input': None,
        },
        {
            'name': 'googlenet-12.onnx',
            'dynamic_input':
                {
                    'data_0': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'inception-v2-9.onnx',
            'dynamic_input': None,
        },
        {
            'name': 'Inceptionv3_rerodered.onnx',
            'dynamic_input':
                {
                    'image': numpy.zeros((1, 3, 299, 299), numpy.float32)
                }
        },
        {
            'name': 'mobilenetv2-12.onnx',
            'dynamic_input':
                {
                    'input': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'resnet50-v1-12.onnx',
            'dynamic_input':
                {
                    'data': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'resnet18-v1-7.onnx',
            'dynamic_input':
                {
                    'data': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'resnet50-v2-7.onnx',
            'dynamic_input':
                {
                    'data': numpy.zeros((1, 3, 224, 224), numpy.float32)
                }
        },
        {
            'name': 'rvm_mobilenetv3_fp32.onnx',
            'dynamic_input':
                {'src': create_ndarray_f32((1, 3, 1080, 1920)), 'r1i': create_ndarray_f32((1, 16, 135, 240)),
                 'r2i': create_ndarray_f32((1, 20, 68, 120)), 'r3i': create_ndarray_f32((1, 40, 34, 60)),
                 'r4i': create_ndarray_f32((1, 64, 17, 30)),
                 'downsample_ratio': numpy.array((0.25,), dtype=numpy.float32)}
        },
        {
            'name': 'shufflenet-v2-12.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 224, 224))
                }
        },
        {
            'name': 'squeezenet1.0-12.onnx',
            'dynamic_input':
                {
                    'data_0': create_ndarray_f32((1, 3, 224, 224))
                }
        },
        {
            'name': 'mobilenetv2-7.onnx',
            'dynamic_input':
                {
                    'data': create_ndarray_f32((1, 3, 224, 224)),
                }
        },
        {
            'name': 'vgg19-7.onnx',
            'dynamic_input': None
        },
        {
            'name': 'yolov4.onnx',
            'dynamic_input':
                {
                    'input_1:0': create_ndarray_f32((1, 416, 416, 3))
                }
        },
        {
            'name': 'tinyyolov2-8.onnx',
            'dynamic_input':
                {
                    'image': create_ndarray_f32((1, 3, 416, 416)),

                }
        },
        {
            'name': 'ssd-12.onnx',
            'dynamic_input':
                {
                    'image': create_ndarray_f32((1, 3, 1200, 1200)),
                }
        },
        {
            'name': 'retinanet-9.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 480, 640)),

                }
        },
        {
            'name': 'ResNet101-DUC-12.onnx',
            'dynamic_input':
                {
                    'data': create_ndarray_f32((1, 3, 800, 800)),
                }
        },
        {
            'name': 'arcfaceresnet100-8.onnx',
            'dynamic_input':
                {
                    'data': create_ndarray_f32((1, 3, 112, 112)),

                }
        },
        {
            'name': 'age_googlenet.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 224, 224)),

                }
        },
        {
            'name': 'emotion-ferplus-8.onnx',
            'dynamic_input':
                {
                    'Input3': create_ndarray_f32((1, 1, 64, 64)),
                }
        },
        {
            'name': 'vgg_ilsvrc_16_age_chalearn_iccv2015.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 224, 224)),

                }
        },
        {
            'name': 'version-RFB-640.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 480, 640)),

                }
        },
        {
            'name': 'bidaf-9.onnx',
            'dynamic_input':
                {
                    'context_word': create_ndarray_f32((16, 1)),
                    'context_char': create_ndarray_f32((16, 1, 1, 16)),
                    'query_word': create_ndarray_f32((16, 1)),
                    'query_char': create_ndarray_f32((16, 1, 1, 16)),

                }
        },
        {
            'name': 'MaskRCNN-12.onnx',
            'dynamic_input':
                {
                    'image': create_ndarray_f32((3, 224, 224)),
                }
        },
        {
            'name': 'gpt2-10.onnx',
            'dynamic_input':
                {
                    'input1': create_ndarray_int64((1, 1, 8)),
                }
        },
        {
            'name': 'roberta-base-11.onnx',
            'dynamic_input':
                {
                    'input_ids': create_ndarray_int64((1, 8)),
                }
        },
        {
            'name': 't5-encoder-12.onnx',
            'dynamic_input':
                {
                    'input_ids': create_ndarray_int64((1, 8)),
                }
        },
        {
            'name': 't5-decoder-with-lm-head-12.onnx',
            'dynamic_input':
                {
                    'input_ids': create_ndarray_f32((1, 8)),
                    'encoder_hidden_states': create_ndarray_f32((1, 8, 768)),

                }
        },
        {
            'name': 'fcn-resnet50-12.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 256, 256)),
                }
        },
        {
            'name': 'FasterRCNN-12.onnx',
            'dynamic_input':
                {
                    'image': create_ndarray_f32((3, 416, 416)),
                }
        },
        {
            'name': 'EdgeNeXt-small.onnx',
            'dynamic_input':
                {
                    'image': create_ndarray_f32((1, 3, 224, 224)),
                }
        },
        {
            'name': 'realesrgan-x4plug.onnx',
            'dynamic_input':
                {
                    'input': create_ndarray_f32((1, 3, 64, 64)),
                }
        },
        {
            'name': 'distilbert_quan80_vnni.onnx',
            'dynamic_input': None
        },
        {
            'name': 'BERT_quan95.onnx',
            'dynamic_input': None
        },
        {
            'name': 'BERT_quant80_vnni.onnx',
            'dynamic_input': None
        },
        {
            'name': 'resnet50_quan75_4blk.onnx',
            'dynamic_input': None
        },
        {
            'name': 'resnet50_quan95.onnx',
            'dynamic_input': None
        },
        {
            'name': 'yolov5s_quanpruned.onnx',
            'dynamic_input': {
                'input': create_ndarray_f32((1, 3, 640, 640))
            }
        },
        {
            'name': 'mobilenetv1_quanpruned_sim.onnx',
            'dynamic_input': None
        },
        {
            'name': 'obert_quan90.onnx',
            'dynamic_input': None
        },
        {
            'name': 'vgg19_quanpruned.onnx',
            'dynamic_input': None
        },
    ]
}
