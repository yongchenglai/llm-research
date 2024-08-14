# owlv2_cli_demo_ensemble.py
"""
python owlv2_cli_demo_ensemble.py \
--print_model \
--model_name_or_path="google/owlv2-large-patch14-ensemble" \
--image_url="http://images.cocodataset.org/val2017/000000039769.jpg"
"""
import requests
from PIL import Image
from PIL import ImageDraw
import numpy as np
import torch
import argparse
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


# Note: boxes need to be visualized on the padded, unnormalized image
# hence we'll set the target image sizes (height, width) based on that
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) \
                         + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--model_name_or_path", type=str,
                        default="google/owlv2-large-patch14-ensemble")
    parser.add_argument(
        "--image_url", type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument('--print_model', action='store_true', default=False,
                        help='print model')
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_name_or_path)

    if args.print_model:
        print(model)

    image = Image.open(requests.get(args.image_url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # Perform a forward pass through the model.
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    unnormalized_image = get_preprocessed_image(inputs.pixel_values)

    # Convert model outputs to COCO API format.
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    # Convert outputs (bounding boxes and class logits)
    # to final bounding boxes and scores
    results = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.2,
        target_sizes=target_sizes
    )

    # Retrieve predictions for the first image
    # for the corresponding text queries
    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], \
                            results[i]["scores"], \
                            results[i]["labels"]


    # Draw bounding boxes and labels on the image.
    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence "
              f"{round(score.item(), 3)} at location {box}")
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")

        draw.text(xy=(x1, y1), text=text[label])

    # Display the image with bounding boxes and labels.
    image.show()


