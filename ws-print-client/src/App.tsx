import React, { useContext, useState } from "react";
import { InferContext } from "./main";
import { Observer, observer } from "mobx-react-lite";
import FrameView from "./components/FrameView";
import { action } from "mobx";
import styled from "styled-components";
import Predictions from "./components/Predictions";
import FrameUpload from "./components/FrameUpload";
import ConnectionStatus from "./components/ConnectionStatus";
import StartStream from "./components/StartStream";

const AppRoot = styled.main`
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: 1fr;
    grid-column-gap: 0px;
    grid-row-gap: 0px;
    width: 100vw;
    min-height: 100vh;
`;

const SideColumn = styled.section`
    display: flex;
    flex-direction: column;
    padding: 15px;
    grid-area: 1/2;
    background-color: #eee;
    padding-left: 5%;
`;

const App = observer(() => {
    const inferStore = useContext(InferContext);
    return (
        <AppRoot>
            <FrameView />
            <SideColumn>
                <ConnectionStatus />
                <FrameUpload />
                <StartStream />
                <div>
                    <Observer>{() => <div>{inferStore.error}</div>}</Observer>
                </div>
                <Predictions />
            </SideColumn>
        </AppRoot>
    );
});

export default App;
