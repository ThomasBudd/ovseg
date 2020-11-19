from ovseg.model.SegementationUNetModel import SegmentationUNetModel


def load_segmentation(is_inference_only=True):
    val_fold = 0
    data_name = 'Task101_OV04BLPOD'
    model_name = 'nnUNet_comparisson_v2'
    return SegmentationUNetModel(val_fold=val_fold, data_name=data_name, model_name=model_name,
                                 is_inference_only=is_inference_only)