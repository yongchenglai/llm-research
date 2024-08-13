# detr-reset-50-basic-demo.py
"""
export HF_ENDPOINT=https://hf-mirror.com
python detr-reset-50-basic-demo.py

python detr-reset-50-basic-demo.py \
--model_name_or_path="facebook/detr-resnet-50" \
--image_url="http://images.cocodataset.org/val2017/000000039769.jpg"
"""
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="facebook/detr-resnet-50")
    parser.add_argument(
        "--image_url", type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    args = parser.parse_args()

    image = Image.open(requests.get(args.image_url, stream=True).raw)
    model_path = args.model_name_or_path

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision="no_timm",
    )

    model = DetrForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=model_path,
        revision="no_timm",
    )

    print(model)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.9)[0]

    for score, label, box in zip(results["scores"],
                                 results["labels"],
                                 results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )


