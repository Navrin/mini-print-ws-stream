import { observer } from "mobx-react-lite";
import { useContext, useEffect, useRef } from "react";
import { InferContext } from "./index";
import { autorun, observable, trace } from "mobx";
import { InferResponseResult, InferState } from "./infer-state";
export interface IFrameViewProps {
  currentImageFrame: Blob;
  imageDim?: [number, number];
  inferData: InferResponseResult[];
}

const FrameView = observer(({ imageDim, inferData }: IFrameViewProps) => {
  const inferStore = useContext(InferContext);
  const currentImageFrame = observable(inferStore.currentImageFrame);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [x, y] = imageDim != null ? imageDim : [0, 0];
  trace(true);
  useEffect(
    () =>
      autorun(() => {
        console.log("effect called");

        if (currentImageFrame.size == 0 || canvasRef == null) return;
        console.log("drawing");
        const canvas = canvasRef.current;
        if (canvas == null) return;
        const ctx = canvas.getContext("2d");
        if (ctx == null) return;

        drawFrame(currentImageFrame, inferData, ctx);
      }),
    [currentImageFrame]
  );

  return currentImageFrame !== null ? (
    <canvas ref={canvasRef} height={y} width={x} />
  ) : (
    <div>no image set...</div>
  );
});

export default FrameView;

async function drawFrame(
  currentImageFrame: Blob,
  inferData: InferResponseResult[],
  ctx: CanvasRenderingContext2D
) {
  const img = await createImageBitmap(currentImageFrame!);

  ctx!.drawImage(img, 0, 0);
  for (const result of inferData) {
    const { x1, y1, y2, x2 } = result.box;

    ctx!.beginPath();
    ctx!.strokeStyle = "red";
    ctx!.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
}
