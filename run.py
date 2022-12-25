""" Run file for camera_perception """

import yaml

from camera_perception import Pipeline


if __name__ == '__main__':
    with open('config/coco.yml', 'r') as f:
        config = yaml.safe_load(f)

    with Pipeline(**config) as p:
        p.run()
