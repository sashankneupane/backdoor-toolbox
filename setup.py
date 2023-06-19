from setuptools import setup, find_packages

setup(
    name='backdoor',
    version='1.0.0',
    author='Sashank Neupane',
    author_email='sn3006@nyu.edu',
    description='Toolbox for backdoor attacks and defenses.',
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files
    package_data={'backdoor': ['poisons/triggers/*.png']},  # Specify the PNG files to include
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
    ],
    python_requires='>=3.6',
)
