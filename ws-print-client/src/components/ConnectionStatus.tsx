import styled from "styled-components";
import React, { useContext } from "react";
import { InferContext } from "../main";
import { darken, desaturate, lighten } from "polished";
import { compose } from "ramda";
import { observer } from "mobx-react-lite";
const warnColor = "#ed6c30a9";
const goodColor = "#1f9c61";
const badColor = "#9c1f4f";
const mapColors = (state) => {
    if (state === WebSocket.CONNECTING) return warnColor;
    if (state === WebSocket.OPEN) return goodColor;
    return badColor;
};
const getBackground = compose(desaturate(0.2), darken(0.1));
const getContrast = compose(lighten(0.5));
const ConnectionStatusBase = styled.div<{ state: number }>`
    display: flex;
    flex-direction: row;
    justify-content: center;
    padding: 5%;
    background-color: ${({ state }) => getBackground(mapColors(state))};
    color: ${({ state }) => getContrast(mapColors(state))};
    border: 4px solid ${({ state }) => getContrast(mapColors(state))};
    font-size: 2rem;
    border-radius: 30px;
    margin-bottom: 20px;
`;
interface IConnectionStatusProps {}

const ConnectionStatus = observer((props: IConnectionStatusProps) => {
    const inferStore = useContext(InferContext);

    const status = inferStore.connectionState;
    if (status == WebSocket.CONNECTING) {
        return (
            <ConnectionStatusBase state={status}>
                Not Connected...
            </ConnectionStatusBase>
        );
    }
    if (status == WebSocket.OPEN) {
        return (
            <ConnectionStatusBase state={status}>Ready!</ConnectionStatusBase>
        );
    }
    return (
        <ConnectionStatusBase state={status}>
            Closed or Unknown
        </ConnectionStatusBase>
    );
});

export default ConnectionStatus;
