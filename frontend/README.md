# Frontend

React + TypeScript + Vite

## Table of Contents
- [Setting up the project](#setting-up-the-project)
- [Running the application](#running-the-application)
- [Linting and dependency management](#linting-and-dependency-management)

## Setting up the project

```shell
# Install frontend dependencies
cd frontend
nvm use
yarn

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env
```

## Running the application

To run the application:

Start the backend:
```
cd backend && PYTHONPATH=.:../chatbot uvicorn main:app --reload
```

Start the frontend (in a new terminal):
```
cd frontend && yarn dev
```

The application will be available at http://localhost:5173, with the backend API at http://localhost:5433.

## Linting and dependency management

Knip finds and fixes unused dependencies, exports and files. Use it for enhanced code and dependency management.

Install it https://knip.dev/overview/getting-started and run it in the frontend directory:

```shell
yarn knip
```

We also installed the following ESLint plugins for better code quality and performance:
- [eslint-plugin-sonarjs](https://github.com/SonarSource/SonarJS/blob/master/packages/jsts/src/rules/README.md) provides a collection of rules to detect bugs and suspicious patterns in your code.
- `eslint-plugin-react-perf` provides React-specific lint rules to improve performance and best practices in React applications.

To run ESLint with the recommended configuration, use:
```shell
yarn lint
```
