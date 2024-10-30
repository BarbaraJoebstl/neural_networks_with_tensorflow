import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from PIL import Image
import argparse
import json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-path", 
        action="store",
        default="assets/Flowers.png",
        dest="image_path",
        type=str,
        help="Path to the image that you want to classify."
        )

    parser.add_argument(
        "--model_-ame", 
        action="store",
        dest="model_name",
        default="my_first_model.h5",
        type=str,
        help="Filename of the model you want to use"
        )

    parser.add_argument(
        "--category-names",
        action="store",
        dest="category_names",
        default='label_map.json',
        type=str,
        help="key value file with the acutal names of the model values"
    )

    parser.add_argument(
        "--top-k",
        action="store",
        dest="top_k",
        default=3,
        type=int,
        help= "Number of the top k most likely classes to be returned."
    )

    print("starting")


    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_name
    category_names = args.category_names
    top_k = args.top_k
    print(image_path, model_path, category_names, top_k)

    print(f"Processing your image {image_path}....")

    class_names = get_class_names(category_names)
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':tfhub.KerasLayer})
    top_k_values, top_k_indices, image = predict(image_path, loaded_model, top_k)
    print_result(top_k_indices, class_names)


def get_class_names(category_names):
    with open('label_map.json', 'r') as f:
        return json.load(f)


def process_image(image):
    image = tf.convert_to_tensor(image, tf.float32)
    # resize
    image = tf.image.resize(image, (224, 224))
    # normalize pixels to range [0,1]
    image /= 255
    return image.numpy()


def predict(path, model, top_k):
    if top_k < 1:
        top_k = 1
    image = Image.open(path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    probes = model.predict(expanded_image)
    top_k_values, top_k_indices = tf.nn.top_k(probes, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_indices = top_k_indices.numpy()
    
    return top_k_values, top_k_indices, image


def print_result(top_k_indices, class_names):
    print("Your image is most likely one of the following:")
    flower_names = []

    for idx in top_k_indices[0]:
        flower_names.append(class_names[str(idx+1)])
    print(flower_names)



if __name__ == '__main__':
    main()