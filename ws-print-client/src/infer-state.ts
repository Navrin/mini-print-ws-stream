import {
    action,
    computed,
    flow,
    makeAutoObservable,
    makeObservable,
    observable,
    runInAction,
} from "mobx";
import OneSignal from "react-onesignal";
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
interface WSSessionConnect {
    kind: "CONNECT_SESSION";
    session_key: string;
}
interface WSNotificationSubscribe {
    kind: "REQUEST_NOTIFICATION";
    email: string | null;
    conditions: [string, number][];
    extern_id: string;
}
type WSMessageBody =
    | WSStopStream
    | WSInferStream
    | WSInferFrame
    | WSSessionStart
    | WSSessionConnect
    | WSNotificationSubscribe;

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
    static SESSION_TOKEN_STORGE = "SESSION_TOKEN";

    _sessionToken: string | null;
    reconnected: boolean = false;

    @computed
    get sessionToken() {
        if (this._sessionToken == null) {
            console.log("session token was nulled");
            const storedToken = localStorage.getItem(
                InferState.SESSION_TOKEN_STORGE,
            );
            console.log(storedToken);
            if (storedToken != "") {
                this._sessionToken = storedToken;
            }
        }
        return this._sessionToken;
    }

    set sessionToken(token: string | null) {
        localStorage.setItem(InferState.SESSION_TOKEN_STORGE, token || "");
        this._sessionToken = token;
        console.log(`saving ${token} into localstorage`);
    }
    connection!: WebSocket;

    inferData: InferResponseResult[];
    lastError = "";
    watchingStream: boolean = false;
    notificationsEnabled: boolean = false;

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
        // this.sessionToken = null;
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
        window.onbeforeunload = () => {
            this.connection.onclose = function () {}; // disable onclose handler first
            this.connection.close();
        };
        this.connection.addEventListener("open", this.onWebConnect);
        this.connection.addEventListener("error", this.handleSocketError);
        this.connection.addEventListener("message", this.onMessage);
        this.connection.addEventListener("close", () => {
            this.connectionState = this.connection.readyState;
            console.log("session terminated");

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
        this.connection.close();

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
        this.connectionState = this.connection.readyState;
        const { body }: WSResponse = JSON.parse(data);

        console.log(`Websocket Message ${body.kind}: 
${JSON.stringify(body, (k, v) => (k == "data" ? JSON.parse(v) : v), 4)}`);

        switch (body.kind) {
            case "SESSION_CONNECTED":
                if (this.sessionToken === body.session_key) {
                    this.reconnected = true;
                    console.log("session reconnected");
                } else {
                    this.sessionToken = body.session_key;
                }
                break;
            case "ERROR":
                if (body.error.includes("session associated")) {
                    console.log("session invalid");
                    this.sessionToken = null;
                    localStorage.removeItem(InferState.SESSION_TOKEN_STORGE);
                    this.onWebConnect();
                } else {
                    this.error = body.error;
                }
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

        this.connectionState = this.connection.readyState;
    }

    onWebConnect = () => {
        this.connectionState = this.connection.readyState;
        console.log(`session token is ${this.sessionToken}`);
        if (this.sessionToken != null) {
            this.connection.send(
                wrapMessage({
                    kind: "CONNECT_SESSION",
                    session_key: this.sessionToken,
                }),
            );
        } else {
            this.connection.send(wrapMessage({ kind: "START_SESSION" }));
        }
        this.connectionState = this.connection.readyState;
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
        console.log(`connstate ${val}`);
        this._connState = val;
    }

    async sendNotificationRequest(formState) {
        const extern_id = OneSignal.User.PushSubscription.id;
        console.log(extern_id);
        const alerts: [string, number][] = Object.entries(
            formState.alerts as {
                [key: string]: { active: boolean; threshold: null | number };
            },
        )
            .filter(([, val]) => val.active)
            .map(([cls, val]) => [cls, 1 / (val.threshold || 80)]);

        this.connection.send(
            wrapMessage({
                kind: "REQUEST_NOTIFICATION",
                conditions: alerts,
                email: formState.email,
                extern_id: extern_id as string,
            }),
        );
    }
}

export { InferState };
