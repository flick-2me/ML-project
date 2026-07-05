from setuptools import find_packages,setup
from typing import List



def get_requirements(file_path:str)->List[str] : 
    requirements = []
    Hypen_Dot = "-e ."
    with open(file_path,"r") as f :
        requirements = f.readlines()
        requirements = [requi.replace("/n","") for requi in requirements ]

        if Hypen_Dot in requirements :
            requirements.remove(Hypen_Dot)
        
        return requirements 



setup(
name= "ml project",
version='0.0.1',
author='Shubham',
author_email='rawatshubham1801@gmail.com',
packages=find_packages(),
install_requires= get_requirements("requirements.txt")
)