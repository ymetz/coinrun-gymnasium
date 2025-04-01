from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='coinrun-gymnasium',
    version='0.0.2',
    description='Gymnasium port of CoinRun: A Reinforcement Learning Environment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yannick Metz, (Original author: OpenAI)',
    url='https://github.com/ymetz/coinrun-gymnasium',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'gymnasium>=1.0.0',
        'pyglet>=2.0.0',
        'joblib>=1.0.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  # Adjust license as needed
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)