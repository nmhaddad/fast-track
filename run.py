import yaml

from camera_perception.pipeline import Pipeline


if __name__ == '__main__':
    with open('config/coco.yml', 'r') as f:
        config = yaml.safe_load(f)

    with Pipeline(config) as p:

        p.run()

