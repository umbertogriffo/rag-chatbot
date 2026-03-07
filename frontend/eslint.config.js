import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import sonarjs from 'eslint-plugin-sonarjs'
import reactPerf from 'eslint-plugin-react-perf'

export default tseslint.config(
    {ignores: ['dist']},
    {
        extends: [
            js.configs.recommended,
            ...tseslint.configs.recommended,
        ],
        files: ['**/*.{ts,tsx}'],
        languageOptions: {
            ecmaVersion: 2020,
            globals: globals.browser,
        },
        plugins: {
            'react-hooks': reactHooks,
            'react-refresh': reactRefresh,
            'sonarjs': sonarjs,
            "react-perf": reactPerf
        },
        rules: {
            ...reactHooks.configs.recommended.rules,
            'sonarjs/no-implicit-dependencies': 'error',
            'react-refresh/only-export-components': [
                'warn',
                {allowConstantExport: true},
            ],
            'react-perf/jsx-no-new-object-as-prop': 'warn',
            'react-perf/jsx-no-new-array-as-prop': 'warn',
            'react-perf/jsx-no-new-function-as-prop': 'warn',
        },
    },
)
