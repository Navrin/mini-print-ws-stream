import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import { createContext } from "react";
import { InferState } from "./infer-state";

const root = ReactDOM.createRoot(
    document.getElementById("root") as HTMLElement,
);

export const InferContext = createContext<InferState>(
    null as unknown as InferState,
);

root.render(
    // <React.StrictMode>
    <InferContext.Provider value={new InferState()}>
        <App />
    </InferContext.Provider>,
    //</React.StrictMode>,
);
// navigator.serviceWorker
//     .register("OneSignalSDKWorker.js")
//     .catch((e) => console.error(e));
