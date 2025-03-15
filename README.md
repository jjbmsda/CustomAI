# Custom AI Model Service

## Overview

This project provides a web-based service that allows users to create, train, and use custom AI models based on their specific needs. 
The backend is powered by **FastAPI**, and the frontend is built using **Streamlit** for an easy-to-use interface.

## Features

- **Data Upload**: Users can upload CSV files for training.
- **Custom Model Training**: Automatically generates and trains a neural network model.
- **Prediction API**: Users can input feature values and get real-time predictions.
- **User-Friendly Interface**: Simple web-based UI built with Streamlit.

## Project Structure

```
custom_ai_service/
│── backend/
│   ├── main.py  # FastAPI server
│   ├── model_handler.py  # Model creation and training
│   ├── requirements.txt  # Backend dependencies
│── frontend/
│   ├── app.py  # Streamlit frontend
│   ├── requirements.txt  # Frontend dependencies
│── README.md  # Project documentation
```

## Installation

### **1. Backend Setup**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### **2. Frontend Setup**

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Endpoints

| Method | Endpoint        | Description                               |
| ------ | --------------- | ----------------------------------------- |
| GET    | `/`             | Check if the API is running               |
| POST   | `/upload-data/` | Upload a CSV file for training            |
| POST   | `/train-model/` | Train a new AI model                      |
| POST   | `/predict/`     | Make a prediction based on input features |

## Usage

1. **Upload Data**: Upload a CSV file containing training data.
2. **Train Model**: Click the "Train Model" button to start training.
3. **Make Predictions**: Input feature values and get predictions in real time.

## Dependencies

### **Backend**

- FastAPI
- Uvicorn
- TensorFlow
- NumPy

### **Frontend**

- Streamlit
- Requests

## Future Enhancements

- **Cloud Deployment**: Deploy on AWS/GCP with a scalable API.
- **Model Selection**: Allow users to choose different types of models (CNN, LSTM, etc.).
- **User Authentication**: Implement authentication for personalized model management.
- **Dockerization**: Add Docker support for easy deployment.

## License

This project is licensed under the MIT License.

---

**Contributions are welcome!!!** Feel free to submit issues and pull requests.


