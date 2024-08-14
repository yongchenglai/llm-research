# owlvit_cli_demo.py
"""
export HF_ENDPOINT=https://hf-mirror.com
python owlvit_cli_demo.py

python owlvit_cli_demo.py \
--model_name_or_path="google/owlvit-large-patch14" \
--image_url="http://images.cocodataset.org/val2017/000000039769.jpg" \
--print_model
"""
import requests
from PIL import Image
import torch
import argparse
from transformers import OwlViTProcessor, OwlViTForObjectDetection


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="google/owlvit-large-patch14")
    parser.add_argument(
        "--image_url", type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument('--print_model', action='store_true', default=False,
                        help='print model')
    args = parser.parse_args()

    processor = OwlViTProcessor.from_pretrained(args.model_name_or_path)
    model = OwlViTForObjectDetection.from_pretrained(args.model_name_or_path)

    if args.print_model:
        print(model)

    image = Image.open(requests.get(args.image_url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.1,
        target_sizes=target_sizes)

    # Retrieve predictions for the first image for the corresponding text queries
    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], \
                            results[i]["scores"], \
                            results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence "
              f"{round(score.item(), 3)} at location {box}")

