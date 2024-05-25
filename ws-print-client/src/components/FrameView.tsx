import { observer } from "mobx-react-lite";
import { useContext, useEffect, useRef, useState } from "react";
import { InferContext } from "../main";
import { autorun, observable, runInAction, trace } from "mobx";
import { InferResponseResult, InferState } from "../infer-state";
import styled from "styled-components";
import React from "react";
export interface IFrameViewProps {}

const FrameContainer = styled.div`
    overflow: hidden;
    grid-area: 1/1;
`;

const FrameView = observer((props: IFrameViewProps) => {
    const inferStore = useContext(InferContext);
    const [imgData, setImageData] = useState<ImageBitmap | null>(null);
    const [inferData, setInferData] = useState<InferResponseResult[]>([]);
    const frameRef = useRef<number>();

    // const img = inferStore.file;

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [x, y] = imgData != null ? [imgData.width, imgData.height] : [0, 0];
    useEffect(() => {
        // console.log("effect called");
        if (inferStore.file == null) return;
        createImageBitmap(inferStore.file).then(setImageData);
        return () => {};
    }, [inferStore.file]);

    useEffect(() => {
        setInferData(inferStore.inferData);
        return () => {};
    }, [inferStore.inferData]);

    useEffect(() => {
        const handle = setInterval(async () => {
            if (inferStore.currentFramePath == "") return;

            try {
                const res = await fetch(inferStore.frameURL);
                const blob = await res.blob();
                const data = await createImageBitmap(blob);
                setImageData(data);
            } catch (e) {
                runInAction(() => {
                    console.error(e);
                    inferStore.lastError = `${e}`;
                });
            }
        }, 33);

        return () => {
            clearInterval(handle);
        };
    }, []);

    const animate = () => {
        // console.log(imgData);
        if (imgData == null || canvasRef.current == null) return;
        // console.log("drawing");
        const canvas = canvasRef.current;
        if (canvas == null) return;
        const ctx = canvas.getContext("2d");
        if (ctx == null) return;

        ctx!.drawImage(imgData, 0, 0);
        for (const result of inferData) {
            const { x1, y1, y2, x2 } = result.box;
            const scaleFactorX = x / inferStore.originalDim[0];
            const scaleFactorY = y / inferStore.originalDim[1];
            ctx!.beginPath();
            ctx!.strokeStyle = "red";
            ctx!.strokeRect(
                x1 * scaleFactorX,
                y1 * scaleFactorY,
                (x2 - x1) * scaleFactorX,
                (y2 - y1) * scaleFactorY,
            );
        }
        frameRef.current = requestAnimationFrame(animate);
    };

    useEffect(() => {
        frameRef.current = requestAnimationFrame(animate);

        return () => cancelAnimationFrame(frameRef.current!);
    });

    return (
        <FrameContainer>
            <canvas ref={canvasRef} height={y} width={x} />
        </FrameContainer>
    );
});

export default FrameView;
