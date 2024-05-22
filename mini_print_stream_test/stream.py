import asyncio
import enum
import json
from json import JSONDecodeError
import base64 as b64
from typing import Literal
import secrets

from pydantic import BaseModel, ConfigDict, ValidationError, Field
from pydantic.dataclasses import dataclass
import os
import websockets
from websockets.server import serve
from ultralytics import YOLO
from ultralytics.engine.results import Results
from statemachine import StateMachine, State

class SessionState(StateMachine):
    ready = State(initial=True)
    get_single_frame = State()
    infer = State()
    inferring_stream = State()

    session_token: str

    def __init__(self, session_token: str):
        self.session_token = session_token
        super(SessionState, self).__init__()

    infer_frame = ready.to(get_single_frame)
    infer_stream = ready.to(inferring_stream)
    frame_recv = get_single_frame.to(infer)
    reset = (inferring_stream.to(ready) | infer.to(ready))




model = YOLO("../yolo_weights/best.pt")

# MessageKind = enum.Enum('WebSocketMessageKind', ['INFER_STREAM', 'INFER_FRAME'])



@dataclass
class SocketMessageInferStream:
    stream_url: str
    kind: Literal['INFER_STREAM'] = 'INFER_STREAM'

@dataclass
class SocketMessageInferFrame:
    kind: Literal['INFER_FRAME'] = 'INFER_FRAME'
    # frame_data: str

@dataclass
class WSMessageStartSession:
    kind: Literal['START_SESSION'] = 'START_SESSION'

class WSMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    body: SocketMessageInferFrame | SocketMessageInferStream | WSMessageStartSession = Field(discriminator='kind')
print(WSMessage.model_json_schema())
@dataclass
class WSResponseInferResult:
    data: str
    kind: Literal['INFER_RESULT'] = 'INFER_RESULT'

@dataclass
class WSResponseSessionConnected:
    session_key: str
    kind: Literal['SESSION_CONNECTED'] = 'SESSION_CONNECTED'
@dataclass
class WSResponseError:
    error: str
    kind: Literal['ERROR'] = 'ERROR'

@dataclass
class WSResponseSendBinary:
    message = 'Awaiting binary frame data...'
    kind: Literal['SEND_BINARY'] = 'SEND_BINARY'

class WSResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    is_error: bool = False
    body: WSResponseSendBinary | WSResponseSessionConnected | WSResponseError | WSResponseInferResult = Field(discriminator='kind')

    @staticmethod
    def from_error(error: Exception):
        return WSResponse(is_error=True, body=WSResponseError(error=str(error)))


SESSIONS = {}

async def process(websocket: websockets.WebSocketClientProtocol):
    connected = {websocket}
    session_key = secrets.token_urlsafe(12)
    session_machine = SessionState(session_key)
    SESSIONS[session_key] = session_machine, connected

    try:
        await websocket.send(WSResponse(
            body=WSResponseSessionConnected(session_key=session_key)
        ).model_dump_json())

        await parse_message(websocket, session_machine)

    except ValidationError as e:
        await websocket.send(WSResponse.from_error(e).model_dump_json())
    except JSONDecodeError as e:
        await websocket.send(WSResponse.from_error(e).model_dump_json())
    except websockets.ConnectionClosedOK:
        pass
    finally:
        del SESSIONS[session_key]


async def parse_message(websocket: websockets.WebSocketClientProtocol, st: SessionState):
    async for message in websocket:
        if type(message) is bytes:
            img_path = f'{st.session_token}-image.jpg'

            with open(img_path, 'wb') as img_f:
                img_f.write(message)

            st.frame_recv()

            results = model(img_path, stream=True)

            for result in results:
                await websocket.send(
                    WSResponse(is_error=False, body=WSResponseInferResult(
                        data=result.tojson(),
                    )).model_dump_json()
                )
            os.remove(img_path)
            st.reset()
            continue

        if type(message) is not bytes and st.current_state == st.get_single_frame:
            await websocket.send(
                WSResponse(is_error=True, body=WSResponseError(error='Frame data should be binary.')).model_dump_json())

            st.reset()
            continue

        parsed: WSMessage = WSMessage.model_validate_json(message)
        if parsed.body.kind == 'INFER_FRAME' and st.current_state == st.ready:
            await websocket.send((WSResponse(is_error=False, body=WSResponseSendBinary())).model_dump_json())
            st.infer_frame()




async def handle(websocket):
    message = await websocket.recv()
    try:
        message = WSMessage.model_validate_json(message)
        if message.body.kind != 'START_SESSION':
            return
        await process(websocket)

    except ValidationError as e:
        await websocket.send(WSResponse(is_error=True, body=WSResponseError(error=str(e))).model_dump_json())
    except websockets.ConnectionClosedOK:
        return


async def main():
    async with serve(handle, "localhost", 8765) as websocket:
        await asyncio.Future()

print(WSMessage.model_json_schema(mode='serialization'))
main_model_schema = WSMessage.model_json_schema(mode='serialization')  # (1)!
print(json.dumps(main_model_schema, indent=2))  # (2)!
asyncio.run(main())
