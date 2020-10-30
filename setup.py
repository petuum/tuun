# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import setuptools


def read_requirements(root, file_path):
    requirements = []
    with open(os.path.join(root, file_path)) as f:
        for line in f:
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if line:
                if not line.startswith("-"):
                    requirements.append(line)
                elif line.startswith("-r"):
                    requirements.extend(
                        read_requirements(root, line.split()[1]))
    return requirements


if __name__ == "__main__":
    submit_to_pypi = int(os.getenv("SUBMIT_TO_PYPI", 0))
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        ]
    if not submit_to_pypi:
        classifiers += [
            "NNI Package :: tuner :: TuunTuner :: tuun.TuunTuner"]
    setuptools.setup(
        name="tuun",
        version=os.getenv("TUUN_VERSION", "0.0.0"),
        author="The Tuun Authors",
        author_email="willie.neiswanger@petuum.com",
        description="Hyperparameter tuning via uncertainty modeling",
        url="https://github.com/petuum/tuun-dev",
        classifiers=classifiers,
        packages=setuptools.find_packages(include=["tuun",
                                                   "tuun.*"]),
        python_requires='>=3.6',
        install_requires=read_requirements("requirements",
                                           "requirements.txt")
    )
