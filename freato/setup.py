from setuptools import find_packages, setup
import os
from glob import glob

package_name = "freato"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="akurtz",
    maintainer_email="akurtz@olin.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "waypoint_follow_server = freato.waypoint_follow_server:main",
            "slam_exploration = freato.slam_exploration:main",
            "test_a_star = freato.test_a_star:main",
        ],
    },
)
