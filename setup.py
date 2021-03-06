from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='pix2pix',
      version="1.0",
      description="Le Wagon project : Image-to-image translation with\
                                        conditional adversarial nets",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/pix2pix-train', 'scripts/pix2pix-predict'],
      zip_safe=False)
