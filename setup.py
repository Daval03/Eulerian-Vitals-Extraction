from setuptools import setup, find_packages

setup(
    name="Code",
    version="0.1",
    packages=find_packages(),
    description="Paquete para procesamiento de imágenes en el TFG",
    author="Aldo",
    author_email="cambroneroaldo@gmail.com",
    
    # Necesario para que Python encuentre los paquetes
    package_dir={"": "."},
    
    # Si tienes dependencias, las puedes listar aquí
    install_requires=[
        "numpy",
        "opencv-python",  # Si usas OpenCV
        # otras dependencias...
    ],
    
    # Esto permite que pytest encuentre los tests
    zip_safe=False,
)