from setuptools import setup, find_packages

setup(
    name="mole_api",
    version="0.1.0",
    packages=find_packages(),          # автоматично знайде utils, pipelines, models, routes
    include_package_data=True,
)
