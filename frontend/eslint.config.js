import typescriptEslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import sonarjs from 'eslint-plugin-sonarjs'
import reactPerf from 'eslint-plugin-react-perf'

export default typescriptEslint.config(
    {
        ignores: ['**/dist/**', '**/node_modules/**', '**/build/**', '**/generated/**'],
        files: ['**/*.{ts,tsx}'],
        extends: [
            ...typescriptEslint.configs.recommended,
        ],
        languageOptions: {
            parserOptions: {
                project: ['./tsconfig.node.json', './tsconfig.app.json'],
                tsconfigRootDir: import.meta.dirname,
            },
        },
        plugins: {
            'react-hooks': reactHooks,
            'react-refresh': reactRefresh,
            'sonarjs': sonarjs,
            "react-perf": reactPerf
        },
        rules: {
            ...reactHooks.configs.recommended.rules,
            ...sonarjs.configs.recommended.rules,
            'react-refresh/only-export-components': [
                'warn',
                {allowConstantExport: true},
            ],
            'react-perf/jsx-no-new-object-as-prop': 'warn',
            'react-perf/jsx-no-new-array-as-prop': 'warn',
            'react-perf/jsx-no-new-function-as-prop': 'warn',
        }
    }
)
