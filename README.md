# Docker Titanic ML Flask App

## Description
Ejemplo básico de desarrollo de modelo de ML en Python y despliegue en contenedores.

## Entrenamiento y modelos
Instalación de dependencias.
```bash
pip3 install -r requirements.txt
```

Comando para entrenar y generar modelos.
```bash
python main.py model
```

Comando para validar funcionamiento de modelo con datos de prueba.
```bash
python main.py test
```

## Despliegue
### Despliegue local (Flask local)
Funcionamiento validado a la fecha: February 14th, 2024.
```bash
pip3 install -r ./requirements.txt
python ./docker/app.py
```

### Ejecución Docker (Flask en contenedor local)
Funcionamiento validado a la fecha: February 14th, 2024.
```bash
docker build -t titanic_ml_app -f ./docker/Dockerfile .
docker run -it -p 5000:5000 titanic_ml_app
```

### Ejecución Heroku Containers (Flask en contenedor remoto)
Funcionamiento validado a la fecha: September 12th, 2021.
```bash
heroku login
heroku container:login
heroku container:push web -a titanic-project
heroku container:release web -a titanic-project
```

### Ejecución Heroku GitHub
Tomar como referencia el siguiente [repositorio](https://github.com/DRodrigo96/e18-ml-titanic-app-heroku-github).
