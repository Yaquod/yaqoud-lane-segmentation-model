import os
from glob import glob
from setuptools import setup, find_packages

package_name = "egolanes_lite_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # Include config files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mhmd Kardosha",
    maintainer_email="user@example.com",
    description="ROS2 package for EgoLanes Lite inference",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "egolanes_lite_node = egolanes_lite_ros2.egolanes_lite_node:main",
            "egolanes_ipm_node = egolanes_lite_ros2.egolanes_ipm_node:main",
            "egolanes_vectorizer_node = egolanes_lite_ros2.egolanes_vectorizer_node:main"
        ],
    },
)
