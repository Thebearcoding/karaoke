import js from '@eslint/js';
import globals from 'globals';

export default [
  {ignores: ['karaoke/web/vendor/**', 'node_modules/**']},
  js.configs.recommended,
  {
    rules: {
      'no-unused-vars': ['error', {ignoreRestSiblings: true}],
    },
  },
  {
    files: ['karaoke/web/**/*.js'],
    languageOptions: {
      globals: {...globals.browser, AudioWorkletProcessor: 'readonly', registerProcessor: 'readonly'},
    },
  },
  {files: ['**/*.mjs'], languageOptions: {globals: globals.node}},
];
