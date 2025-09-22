from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'transform_to_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/transform_to_map_launch.py']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'detection_images'), glob('detection_images/*') if os.path.exists('detection_images') else []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dennis',
    maintainer_email='mcde1011@h-ka.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'transform_to_map_node = transform_to_map.transform_to_map_node:main',
        'tf_publisher = transform_to_map.tf_publisher:main'
    ],
    },
)
