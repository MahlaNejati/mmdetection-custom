
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot,multi_gpu_test, single_gpu_test
from mmdet.core import evaluation
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core.evaluation import coco_utils 
import glob
import matplotlib.pyplot as plt
import cv2
from mmdet.core import encode_mask_results
import numpy as np
import datetime as dt



def result_to_json(dataset, results):
    result_files = dict()
    coco = []
    coco = {
        'info': None,
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
        }   

    coco['info'] = {
            'year': dt.datetime.now(dt.timezone.utc).year,
            'version': None,
            'description': "mmdetection_result",
            'contributor': "mnej691",
            'url': "",
            'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
        }

    bbox_json_results = []
    segm_json_results = []
    print(results[0])
    print("*********************************88")
    print(results[1])
    print("here")
    for idx in range(len(results)):
        print(idx)
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        image = {
            "id": data['ID'],
            "width": 0,
            "height": 0,
            "file_name": data['Labeled Data'],
            "license": None,
            "flickr_url": data['Labeled Data'],
            "coco_url": data['Labeled Data'],
            "date_captured": None,
        }

        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            if(len(bboxes)> 0):
                width, height = seg[label].shape
                image["width"] = width
                image["height"] = height
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = coco_utils.xyxy2xywh(bboxes)
                data['score'] = float(bboxes[4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    
    result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
    mmcv.dump(json_results[1], result_files['segm'])
    return bbox_json_results, segm_json_results    


def main():
    config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
    checkpoint = 'tomato/laboro_tomato_48ep.pth'
    images_path = glob.glob('tomato/Outdoor/*')
    # Use the detector to do inference
    images = []
    results = tuple()
    output = 'tomato/tomato_test.json'
    model = init_detector(config, checkpoint, device='cuda:0')
    for img_path in images_path:
        img = cv2.imread(img_path)
        images.append(img)

    result = inference_detector(model, images)
    #results = [(bbox_results, encode_mask_results(mask_results))
    #                  for bbox_results, mask_results in result]

    result_to_json(images, result)

    for i in range(0,len(images)):
        img_result = show_result_pyplot(model, images[i], result[i])


    

if __name__ == "__main__":
    main()


'''old function
############################Input#######################
# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'tomato/laboro_tomato_48ep.pth'
# initialize the detector
images_path = glob.glob('tomato/Outdoor/*')
# Use the detector to do inference
images = []
results = tuple()
output = 'tomato/tomato_test.json'


model = init_detector(config, checkpoint, device='cuda:0')
cfg = mmcv.Config.fromfile(config)
cfg.model.train_cfg = None

samples_per_gpu = 1
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

i = 0
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)
    print("*****************************************")
    print(img_path)
    #bbox_results, mask_results = result
    #print(encode_mask_results(mask_results))
    #result = inference_detector(model, img)
    #if isinstance(result, tuple):
    #    result = [(bbox_results, encode_mask_results(mask_results))
    #                for bbox_results, mask_results in result]
    i = i+1
    if(i >1):
        break
    #print(result)

result = inference_detector(model, images)
results = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
result_to_json(dataset, result, output)

coco_utils.results2json(dataset, results, output)

#mmcv.dump(results, file_format='json')

#cfg = mmcv.Config.fromfile(config)
#print (cfg.data.test)
#cfg.data.test['type'] = 'Test'
#cfg.data.test['ann_file']
#cfg.data.test['img_prefix'] =



#model = init_detector(config, checkpoint, device='cuda:0')
#model = MMDataParallel(model, device_ids=[0])
#outputs = single_gpu_test(model, data_loader)





#checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

#print(model.CLASSES)

#print(outputs)
results = []
#for img_path in images_path:
#    img = cv2.imread(img_path)


#if isinstance(result[0], tuple):
#        results = [(bbox_results, encode_mask_results(mask_results))
#                    for bbox_results, mask_results in result]

'''
'''bbox_result, mask_result = result[0]
print("*****************************************")
print(bbox_result)

print("*****************************************")
labels = [np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)]

print(labels)

print("*****************************************")
labels = np.concatenate(labels)


    bbox_results, mask_results = res
    print ("************************************")
    print (bbox_results)
    print (mask_results)

print(labels)'''

'''
for res in results:
    print ("************************************")
    print (res)

    if isinstance(result[0], tuple):
        res_temp = [(bbox_results, encode_mask_results(mask_results))
                    for bbox_results, mask_results in res]
    results.extend(res_temp)
'''
#mmcv.dump(result, output, file_format='json')

'''
#results.extend(result)




kwargs = {} if args.eval_options is None else args.eval_options
if args.format_only:
    dataset.format_results(outputs, **kwargs)
if args.eval:
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)
    metric_dict = dict(config=args.config, metric=metric)
    if args.work_dir is not None and rank == 0:
        mmcv.dump(metric_dict, json_file)
'''


# build the model and load checkpoint

#model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))





#for i in range(0,len(images)):
#    img_result = show_result_pyplot(model, images[i], results[i],wait_time=3.0)




# visualize the results in a new window
#show_result(img, result)
# or save the visualization results to image files

#show_result(img, result, out_file='result.jpg')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

	
# Let's plot the result
#	show_result_pyplot(model, img, result, 0.3, name, 20)


