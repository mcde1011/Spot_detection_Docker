from glob import glob
from setuptools import setup

package_name = 'ricoh_theta_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob('resource/*.pt')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dennis',
    maintainer_email='',
    description='Ricoh Theta X + ROS2',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ricoh_publisher = ricoh_theta_ros.ricoh_publisher:main',
            'yolo_detector  = ricoh_theta_ros.yolo_detector:main',
            'mosaic_viewer  = ricoh_theta_ros.mosaic_viewer:main',
        ],
    },
)

