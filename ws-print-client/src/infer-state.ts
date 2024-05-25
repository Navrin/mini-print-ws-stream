import {
    action,
    computed,
    flow,
    makeAutoObservable,
    makeObservable,
    observable,
    runInAction,
} from "mobx";
import { StringLiteral } from "typescript";

interface WSResponseSendBinary {
    kind: "SEND_BINARY";
    message: string;
}

interface WSResponseError {
    kind: "ERROR";
    error: string;
}

interface WSResponseSessionConnected {
    session_key: string;
    kind: "SESSION_CONNECTED";
}
interface WSResponseInferResult {
    data: string;
    image_shape: [number, number];
    kind: "INFER_RESULT";
}
interface WSResponseStreamEstablished {
    kind: "STREAM_ESTABLISHED";
    frame_url: string;
}
interface WSResponse {
    is_error: boolean;
    body:
        | WSResponseError
        | WSResponseInferResult
        | WSResponseSendBinary
        | WSResponseSessionConnected
        | WSResponseStreamEstablished;
}

interface WSInferStream {
    kind: "INFER_STREAM";
    stream_url: string;
}
interface WSInferFrame {
    kind: "INFER_FRAME";
}
interface WSSessionStart {
    kind: "START_SESSION";
}
interface WSStopStream {
    kind: "STOP_STREAM";
}

type WSMessageBody =
    | WSStopStream
    | WSInferStream
    | WSInferFrame
    | WSSessionStart;
interface WSMessage {
    body: WSMessageBody;
}
export interface InferResponseResult {
    name: string;
    class: number;
    confidence: number;
    box: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
    };
}

function wrapMessage(msg: WSMessageBody) {
    return JSON.stringify({
        body: msg,
    });
}

class InferState {
    sessionToken: string | null;
    connection!: WebSocket;

    inferData: InferResponseResult[];
    lastError = "";
    watchingStream: boolean = false;
    get error() {
        return this.lastError;
    }
    set error(v: string) {
        this.lastError = v;
    }
    imageDim: [number, number] = [0, 0];
    originalDim: [number, number] = [0, 0];

    @computed get ready() {
        return this.sessionToken != null;
    }

    currentFramePath = "";

    get frameURL() {
        return InferState.FRAME_URL + this.currentFramePath;
    }

    listeners = [];
    static WS_TARGET = "ws://localhost:8765";
    static FRAME_URL = "http://127.0.0.1:5000";
    constructor() {
        this.sessionToken = null;
        this.inferData = [];
        makeAutoObservable(
            this,
            {
                inferData: observable.deep,
            },
            { autoBind: true },
        );
        this.connect();
    }

    private connect() {
        this.connection = new WebSocket(InferState.WS_TARGET);
        this.connectionState = this.connection.readyState;

        this.connection.addEventListener("open", this.onWebConnect);
        this.connection.addEventListener("error", this.handleSocketError);
        this.connection.addEventListener("message", this.onMessage);
        this.connection.addEventListener("close", () => {
            this.connectionState = this.connection.readyState;

            // this.connection.removeEventListener("open", this.onWebConnect);
            // this.connection.removeEventListener("error", this.handleSocketError);
            // this.connection.removeEventListener("message", this.onMessage);
        });
    }

    socketError() {
        runInAction(() => {
            this.error = `There was an error connecting to the websockets server (is the server online, and is it reachable at ${InferState.WS_TARGET}?)`;
        });
        console.log(this);
    }

    resetConnection = () => {
        this.connect();
    };

    handleSocketError = (ev: Event) => {
        this.connectionState = this.connection.readyState;

        console.log("In socket error");
        this.socketError();
        setTimeout(this.resetConnection, 15000);
    };

    @action
    onMessage({ data }: MessageEvent) {
        const { body }: WSResponse = JSON.parse(data);

        console.log(`Websocket Message ${body.kind}: 
${JSON.stringify(body, (k, v) => (k == "data" ? JSON.parse(v) : v), 4)}`);

        switch (body.kind) {
            case "SESSION_CONNECTED":
                this.sessionToken = body.session_key;
                break;
            case "ERROR":
                this.error = body.error;
                break;
            case "INFER_RESULT": {
                const result = JSON.parse(body.data);
                this.inferData = [...result];
                this.originalDim = body.image_shape;
                break;
            }
            case "SEND_BINARY":
                if (this.file === null) {
                    this.error =
                        "No image has been set but client requested infer results!";
                    break;
                }
                this.connection.send(this.file);
                break;
            case "STREAM_ESTABLISHED":
                this.currentFramePath = body.frame_url;
                this.watchingStream = true;
        }
    }

    onWebConnect = () => {
        this.connectionState = this.connection.readyState;
        this.connection.send(wrapMessage({ kind: "START_SESSION" }));
    };

    requestFrameInfer() {
        this.connection.send(
            wrapMessage({
                kind: "INFER_FRAME",
            }),
        );
    }
    @observable _file: File | Blob | null = null;
    get file(): File | Blob | null {
        return this._file;
    }
    set file(f: File | Blob | null) {
        if (f == null) return;
        this._file = f;
    }
    @action
    updateImageDim(bitmap: ImageBitmap) {
        if (this.file == null) return;

        this.imageDim = [bitmap.width, bitmap.height];
    }

    @action
    startStream(streamLink: string) {
        this.connection.send(
            wrapMessage({
                kind: "INFER_STREAM",
                stream_url: streamLink,
            }),
        );
    }
    @action
    stopStream() {
        this.connection.send(
            wrapMessage({
                kind: "STOP_STREAM",
            }),
        );
        this.watchingStream = false;
        this.currentFramePath = "";
    }

    _connState: number = -1;

    get connectionState() {
        return this._connState;
    }

    set connectionState(val: number) {
        this._connState = val;
    }
}

export { InferState };
