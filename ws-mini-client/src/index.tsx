import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import {createContext} from 'react';
import { InferState } from './infer-state';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

export const InferContext = createContext<InferState>(new InferState());

root.render(
  <React.StrictMode>
    <InferContext.Provider value={new InferState()}>
      <App />
    </InferContext.Provider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
