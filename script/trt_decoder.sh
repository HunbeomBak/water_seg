/usr/src/tensorrt/bin/trtexec \
--onnx=runs/241115_SAM_ViT_b_dataset_V2_ft_v1/water_seg_vit_b_decoder.onnx \
--saveEngine=Weights/engine/water_seg_vit_b_decoder.engine \
--minShapes=point_coords:1x1x2,point_labels:1x1 \
--optShapes=point_coords:1x1x2,point_labels:1x1 \
--maxShapes=point_coords:1x10x2,point_labels:1x10