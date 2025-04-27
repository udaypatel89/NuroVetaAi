# Healthcare-AI-WebApp

![slide1](https://drive.google.com/file/d/1VSkInLKezU39x4PFLAOX7pePlKyHHGmd/view?usp=sharing)

## AltairCare app can be accessed [here](https://altaircare.onrender.com)

Artificial intelligence is a rapidly developing field with new improvements that happen every day. It allows the automation and simplification of human activities in different industries like agriculture, banking and financial services, healthcare, travel, and more.

The integration of artificial intelligence in healthcare is particularly appealing to companies like Microsoft, Google, Apple, and IBM. It simplifies the lives of patients, doctors, and hospital administrators by performing tasks typically done by humans, but in less time and at a fraction of the cost. It also enables medical services to be provided in remote and hard-to-reach areas, and can diagnose diseases in their early stages, thus increasing the chances of recovery.

In my application, AltairCare, I used machine learning and deep learning algorithms to train highly accurate models that allow users to check their chances of having one of the following diseases:

- Liver Disease
- Pneumonia Disease
- Kidney Disease
- Diabetes Disease
- Stroke Disease
- Heart Disease

![slide2](https://drive.google.com/file/d/1cmMXL3MK178oBhdHvg-ooPDIR8tsbWyc/view?usp=sharing)

Each disease has its own page with an overview and symptoms that patients may experience, as well as prediction model information and the required parameters that users must provide.

## Disease Page:

![Disease Page](https://drive.google.com/file/d/1VSkInLKezU39x4PFLAOX7pePlKyHHGmd/view?usp=sharing)

## Prediction Page:

![Prediction Page](https://drive.google.com/file/d/1B0ghvXqwZmPEOTYCrQHzurljgL5zTyOF/view?usp=sharing)

All the datasets used to train the models can be found on the Kaggle website.

## Libraries Used:

- **Flask:** for backend web development
- **Scikit-learn & TensorFlow:** for training the disease prediction models
- **SQLAlchemy:** library for handling SQLite database

## How to Run This Project:

### Clone this Repository:

```sh
git clone https://github.com/kaymen99/Healthcare-AI-WebApp.git
cd Healthcare-AI-WebApp
```

### Using Python Directly:

#### Install Requirements (using a virtual environment is preferable):

```sh
pip install -r requirements.txt
```

#### Run this Command to Start the Local Server:

```sh
python app.py
```

### Using Docker:

Build the Docker image (make sure to install Docker Desktop: [Docker Desktop](https://www.docker.com/products/docker-desktop/)):

```sh
docker build -t healthcare-ai-webapp .
```

Run the Docker container:

```sh
docker run -p 5000:5000 healthcare-ai-webapp
```

You should be able to access your app by visiting [http://localhost:5000/](http://localhost:5000/) in your browser.# NuroVetaAi
# NuroVetaAi
# NuroVetaAi
