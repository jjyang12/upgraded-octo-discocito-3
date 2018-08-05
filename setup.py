from setuptools import setup, find_packages

def do_setup():
    setup(name='facerec',
          version="0.0",
          author='upgraded-octo-discocito-3',
          description="it's time to FACE the truth...(EMOTIONAL) (*NOT CLICKBAIT*)",
          license='Cog*Works',
          platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
          packages=find_packages(),
          install_requires=['numpy>=1.12'])

if __name__ == "__main__":
    do_setup()
