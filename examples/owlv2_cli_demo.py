# owlv2_cli_demo.py
"""
python owlv2_cli_demo.py \
--print_model \
--model_name_or_path="google/owlv2-large-patch14-finetuned" \
--image_url="http://images.cocodataset.org/val2017/000000039769.jpg"
"""
import requests
from PIL import Image
import torch
import argparse
from transformers import Owlv2Processor, Owlv2ForObjectDetection


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="google/owlv2-large-patch14-finetuned")
    parser.add_argument(
        "--image_url", type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument('--print_model', action='store_true', default=False,
                        help='print model')
    args = parser.parse_args()

    processor = Owlv2Processor.from_pretrained(args.model_name_or_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name_or_path)

    if args.print_model:
        print(model)

    image = Image.open(requests.get(args.image_url, stream=True).raw)

    # Define the text queries that you want the model to detect.
    # texts = [['face', 'rocket', 'nasa badge', 'star-spangled banner']]
    texts = [["a photo of a cat", "a photo of a dog"]]
    # Prepare the image and text for the model using the processor.
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to
    # rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.1,
        target_sizes=target_sizes)

    # Retrieve predictions for the first image
    # for the corresponding text queries
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


