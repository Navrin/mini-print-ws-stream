import React, { useContext, useState } from "react";

import styled from "styled-components";
import { InferContext } from "../main";
import { action } from "mobx";

const StartStreamBase = styled.div`
    display: flex;
    flex-direction: column;
`;

const StreamInput = styled.input`
    all: unset;
    display: flex;
    width: calc(100% - 20px);
    height: 60px;
    flex-grow: 1;
    border-radius: 30px;
    box-shadow: none;
    outline-color: none;
    border: 2px solid hsl(0, 0%, 40%);
    padding-left: 20px;
    margin-bottom: 5px;
`;
const StreamInputLabel = styled.label`
    padding: 2% 5%;
    font-size: 1.25rem;
`;
const StreamStartButton = styled.button`
    margin: 10px 0px;
    font-size: 1.5rem;
`;

interface IStartStreamProps {}

const StartStream = (props: IStartStreamProps) => {
    const inferStore = useContext(InferContext);
    const [streamLink, setStreamLink] = useState<string>("");
    return (
        <StartStreamBase>
            <StreamInputLabel htmlFor="streamURL">
                URL to any camera stream, can be RTSP, IP/TCP based or
                physical/on-device by using a number like "0".
            </StreamInputLabel>
            <StreamInput
                name="streamURL"
                placeholder="Add stream URL (any opencv format)"
                onChange={(e) => setStreamLink(e.target.value)}
            />
            <StreamStartButton
                disabled={streamLink == null || streamLink === ""}
                onClick={action((e) => {
                    if (streamLink == null) return;
                    e.preventDefault();
                    inferStore.startStream(streamLink);
                })}
            >
                Start Watching
            </StreamStartButton>
        </StartStreamBase>
    );
};

export default StartStream;
