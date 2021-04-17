from setuptools import setup, find_packages

setup(
  name='threebody',
  version='0.1.1',
  description='A python package to visualize three body problem in 3D',
  url='https://github.com/LauraElKaim/Project_Three_Body_Problem.git',
  author='Mohamed Fattouhy; Amine Touzani; Gueladio Niasse; Laura El Kaim',
  packages=find_packages(include=['threebody',
                                  'threebody.EDO',
                                  'threebody.Vis'])
)
