# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config({
  extends: [
    // Remove ...tseslint.configs.recommended and replace with this
    ...tseslint.configs.recommendedTypeChecked,
    // Alternatively, use this for stricter rules
    ...tseslint.configs.strictTypeChecked,
    // Optionally, add this for stylistic rules
    ...tseslint.configs.stylisticTypeChecked,
  ],
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
```


Create a new React TypeScript project:
```shell
npm create vite@latest chatbot-ui -- --template react-ts
cd chatbot-ui
npm cache clean --force

# Disable progress bar
npm config set progress false
# Install yarn globally if not installed
npm install -g yarn
# Remove the existing dependencies and npm lockfile
rm -rf node_modules package-lock.json
# Initialize yarn and install dependencies using your existing package.json
yarn
# Update the scripts in package.json to use yarn instead of npm.
# Installing dependencies
yarn                    # replaces npm install
yarn add [package]      # replaces npm install [package]
yarn add -D [package]   # replaces npm install -D [package]
# Running scripts
yarn dev               # replaces npm run dev
yarn build             # replaces npm run build

# Upgrading dependencies
yarn upgrade           # replaces npm update
```

```shell
# setup.sh
#!/bin/bash

# Install Python dependencies
pip install fastapi uvicorn

# Install frontend dependencies
cd chatbot-ui
yarn

# Create .env file
echo "VITE_API_URL=http://localhost:5433" > .env
```

To run the application:
Start the backend:
```
cd chatbot
uvicorn main:app --reload
```

Start the frontend (in a new terminal):
```
cd chatbot-ui
yarn dev
```

The application will be available at http://localhost:5173, with the backend API at http://localhost:5433.
