import { observer } from "mobx-react-lite";
import React, { useContext } from "react";
import styled from "styled-components";
import { InferContext } from "../main";

const PredictionsBase = styled.div``;
const PredictionHeader = styled.h3``;
const PredictionEntry = styled.span`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
`;
const PredictionClass = styled.span`
    font-size: 2rem;
    font-weight: 600;
`;
const PredictionValue = styled.span`
    font-size: 2rem;
`;
interface IPredictionProps {}

const Predictions = observer((props: IPredictionProps) => {
    const inferStore = useContext(InferContext);

    return (
        <PredictionsBase>
            <PredictionHeader>
                Predictions ({inferStore.inferData.length})
            </PredictionHeader>
            {inferStore.inferData.map((pred) => (
                <PredictionEntry>
                    <PredictionClass>{pred.name}</PredictionClass>
                    <PredictionValue>
                        {(pred.confidence * 100).toFixed(4)}%
                    </PredictionValue>
                </PredictionEntry>
            ))}
        </PredictionsBase>
    );
});

export default Predictions;
