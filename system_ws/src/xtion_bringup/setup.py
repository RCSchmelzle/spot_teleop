from setuptools import setup

package_name = 'xtion_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/multi_xtion.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/xtion_cameras.yaml',
        ]),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='teleop',
    maintainer_email='you@example.com',
    description='Bringup utilities for ASUS Xtion cameras',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gen_xtion_cameras = xtion_bringup.gen_xtion_cameras:main',
            'xtion_mapper_gui = xtion_bringup.xtion_mapper_gui:main',
        ],
    },
)
