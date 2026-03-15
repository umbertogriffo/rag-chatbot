# Backend Setup Instructions

To run the application:
Start the backend:
```shell
cd backend && PYTHONPATH=.:../chatbot uvicorn main:app --reload
```

Start the frontend (in a new terminal):
```shell
cd frontend && yarn dev
```

The application will be available at http://localhost:5173, with the backend API at http://localhost:8000.
