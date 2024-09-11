from setuptools import setup, find_packages

setup(
    name="market_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "asyncio",
        # Add other dependencies here
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
        ],
    },
)
