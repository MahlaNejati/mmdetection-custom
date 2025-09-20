# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'tomato/laboro_tomato_48ep.pth'
# initialize the detector
print("there")
model = init_detector(config, checkpoint, device='cuda:0')
print("here")
# Use the detector to do inference
img = 'tomato/IMG_1003.jpg'
result = inference_detector(model, img)

# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
