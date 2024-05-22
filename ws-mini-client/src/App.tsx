import React, { useContext, useEffect } from "react";
import logo from "./logo.svg";
import "./App.css";
import { InferContext } from "./index";
import { Observer, observer } from "mobx-react-lite";
import FrameView from "./FrameView";
import { action } from "mobx";

function App() {
  const inferStore = useContext(InferContext);
  return (
    <div className="App">
      <FrameView
        currentImageFrame={inferStore.currentImageFrame}
        inferData={inferStore.inferData}
        imageDim={inferStore.imageDim}
      />
      <input
        type="file"
        accept="image/jpg"
        onChange={action((e) => {
          e.preventDefault();
          if (e.target == null || e.target.files == null) return;
          const file = e.target.files[0];

          inferStore.file = file;
        })}
      />
      <button
        onClick={() => {
          inferStore.requestFrameInfer();
        }}
        disabled={!inferStore.ready}
      >
        Request Infer
      </button>
      <div>
        <Observer>{() => <div>{inferStore.error}</div>}</Observer>
      </div>
    </div>
  );
}

export default observer(App);
