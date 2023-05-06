import setuptools

setuptools.setup(
    name="src",
    author="Jüri Keller",
    author_email='jueri.keller@smail.th-koeln.de',
    packages=setuptools.find_packages(
        include=["src"]
    ),
    python_requires=">=3.8",
)