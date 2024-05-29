import uuid
from onesignal.model.notification import Notification
from onesignal.model.player_notification_target_include_aliases import PlayerNotificationTargetIncludeAliases
from onesignal.model.string_map import StringMap
from statemachine import StateMachine, State
from ultralytics.engine.results import Results
from ultralytics import YOLO
import os
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, ValidationError, Field
from contextlib import asynccontextmanager
import functools
import logging
import asyncio
import contextlib
import enum
import json
from json import JSONDecodeError
import base64 as b64
import math
from re import I
import time
from typing import Literal, Optional
import secrets
import typing
import filetype
from flask import session
import trio_asyncio
from sympy import O
import trio
import trio_websocket as ws
import onesignal
from onesignal.api import default_api
import cv2
import nest_asyncio
import dotenv
nest_asyncio.apply()

dotenv.load_dotenv("../.env")

configuration = onesignal.Configuration(
    app_key=os.getenv("ONESIGNAL_KEY"),
    # api_key=os.getenv("ONESIGNAL_KEY")
)

# import websockets
# from websockets.server import serve


model = YOLO("../yolo_weights/best.pt")

# MessageKind = enum.Enum('WebSocketMessageKind', ['INFER_STREAM', 'INFER_FRAME'])


@dataclass
class SocketMessageInferStream:
    stream_url: str
    kind: Literal['INFER_STREAM'] = 'INFER_STREAM'


@dataclass
class WSMessageRequestNotification:
    conditions: list[tuple[str, float]]
    email: str | None
    extern_id: str
    kind: Literal['REQUEST_NOTIFICATION'] = 'REQUEST_NOTIFICATION'


@dataclass
class SocketMessageInferFrame:
    kind: Literal['INFER_FRAME'] = 'INFER_FRAME'
    # frame_data: str


@dataclass
class WSMessageStartSession:
    kind: Literal['START_SESSION'] = 'START_SESSION'


@dataclass
class WSMessageConnectSession:
    session_key: str
    kind: Literal['CONNECT_SESSION'] = 'CONNECT_SESSION'


@dataclass
class WSMessageStopStream:
    kind: Literal['STOP_STREAM'] = 'STOP_STREAM'


class WSMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    body: SocketMessageInferFrame | SocketMessageInferStream | WSMessageStartSession | WSMessageStopStream | WSMessageConnectSession | WSMessageRequestNotification = Field(
        discriminator='kind')


@dataclass
class WSResponseInferResult:
    data: str
    image_shape: tuple[int, int]
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
class WSResponseNotification:
    kind: Literal['NOTIFICATION_CREATED'] = 'NOTIFICATION_CREATED'


@dataclass
class WSResponseSendBinary:
    message = 'Awaiting binary frame data...'
    kind: Literal['SEND_BINARY'] = 'SEND_BINARY'


@dataclass
class WSResponseStreamEstablished:
    frame_url: str
    kind: Literal['STREAM_ESTABLISHED'] = 'STREAM_ESTABLISHED'


class WSResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    is_error: bool = False
    body: WSResponseSendBinary | WSResponseSessionConnected | WSResponseError | WSResponseInferResult | WSResponseStreamEstablished = Field(
        discriminator='kind')

    @staticmethod
    def from_error(error: Exception):
        return WSResponse(is_error=True, body=WSResponseError(error=str(error)))


# logging.getLogger("utils.general").setLevel(logging.WARNING)  # yolov5 logger


class SessionState(StateMachine):
    ## States ##
    ready = State(initial=True)
    get_single_frame = State()
    infer = State()
    inferring_stream = State()
    session_terminated = State(final=True)

    ## Transitions ##
    infer_frame = ready.to(get_single_frame)
    infer_stream = ready.to(inferring_stream)
    frame_recv = get_single_frame.to(infer)
    reset = (inferring_stream.to(ready) | infer.to(
        ready) | get_single_frame.to(ready) | ready.to(ready))

    terminate = (
        ready.to(session_terminated) |
        get_single_frame.to(session_terminated) |
        infer.to(session_terminated) |
        inferring_stream.to(session_terminated)
    )

    def on_enter_terminate(self):
        if self.cancel_scope is not None:
            self.cancel_scope.shield = False
            self.cancel_scope.cancel()

    cap: cv2.VideoCapture
    session_token: str
    connections: set[ws.WebSocketConnection] = set()
    # stream_task: set[asyncio.Task] = set()
    # message_queue = asyncio.Queue()
    stream_target:  str | None = None
    session_dead: float | None = None
    task_nursery: trio.Nursery | None = None
    cancel_scope: trio.CancelScope | None = None
    loop: trio_asyncio.TrioEventLoop | None = None

    send: trio.MemorySendChannel | None = None
    recv: trio.MemoryReceiveChannel | None = None

    def add_connection(self, websocket: ws.WebSocketConnection):
        if websocket in self.connections:
            print('socket already connected')
            return  # raise error?
        session_dead = None
        self.connections.add(websocket)

    async def remove_connection(self, websocket: ws.WebSocketConnection):
        if websocket not in self.connections:
            raise StateInvariantError(f"Connection {
                                      websocket._id} was marked for removal but isn't part of the session!")

        if len(self.connections) == 1:
            self.session_dead = time.time()
        self.connections.remove(websocket)
        await websocket.aclose()

    def set_channel(self, send: trio.MemorySendChannel, recv: trio.MemoryReceiveChannel):
        self.send = send
        self.recv = recv

    async def send_message(self, msg: WSMessage):
        if self.send is None:
            raise StateInvariantError("No send channel!")

        await self.send.send(msg)

    @property
    def frame_name(self):
        return f'{self.session_token}.jpg'

    @property
    def frame_path(self):
        return f'./static/{self.frame_name}'

    def __init__(self, session_token: str):
        self.session_token = session_token
        super(SessionState, self).__init__()

    def end_stream(self):
        self.stream_target = None
        self.cap = None

    def cleanup(self):
        self.reset()
        try:
            os.remove(self.frame_path)

            self.end_stream()
        except (StreamNotValidError, FileNotFoundError):
            pass

    @dataclass
    class NotificationObject:
        conditions: list[tuple[str, float]]
        email: str
        extern_id: str
        last_send: int = -1

    notifications: list[NotificationObject] = []

    def add_notification(self, extern_id, conditions, email=None):
        self.notifications.append(self.NotificationObject(
            conditions=conditions,
            email=email,
            extern_id=extern_id,
        ))

    async def check_notifications(self, result: Results):
        box_cls = {}

        for box in result.summary():
            box_cls[box['name']] = box['confidence']

        for notification in self.notifications:
            if abs(notification.last_send - time.time()) <= 20:
                continue
            for [cls, val] in notification.conditions:
                if cls not in box_cls:
                    continue
                if box_cls[cls] < val:
                    continue

                notification.last_send = time.time()
                with onesignal.ApiClient(configuration) as api_client:
                    # Create an instance of the API class
                    api_instance = default_api.DefaultApi(api_client)
                    notification_msg = Notification(
                        app_id=os.getenv("ONESIGNAL_APP"),
                        # include_external_user_ids=[notification.extern_id],
                        # target_channel="push"
                    )
                    # notification_msg.set_attribute(
                    # 'external_id', str(uuid.uuid4()))
                    contentsStringMap = StringMap()

                    contentsStringMap.set_attribute(
                        'en', f"[Print Alert] {cls} at {box_cls[cls]}!")

                    notification_msg.set_attribute(
                        'contents', contentsStringMap)
                    notification_msg.set_attribute('is_any_web', True)
                    # notification_msg.set_attribute('is_any_web', True)
                    notification_msg.set_attribute(
                        'include_subscription_ids', [notification.extern_id])

                    try:
                        print("sending out notification!")
                        api_response = api_instance.create_notification(
                            notification_msg)
                        res = api_response
                        print(f'api_resonse was {res=}')

                    except onesignal.ApiException as e:
                        print(f"encounted onesignal error {e=} {
                            e.reason} {e.status} {e.body}")


SESSIONS: dict[str, SessionState] = {}


async def process(websocket: ws.WebSocketConnection, reconnect_session_key=None):
    ###
    # TODO: Change the following acception to handle
    # reconnects using a session key
    ###
    if reconnect_session_key is None:
        session_key = secrets.token_urlsafe(12)
        session_machine = SessionState(session_key)
        session_machine.add_connection(websocket)

        SESSIONS[session_key] = session_machine
    elif reconnect_session_key not in SESSIONS:
        print(f'session key {reconnect_session_key} not found')

        raise SessionNotFoundError(
            f'{reconnect_session_key} does not have a session associated with it anymore.')

    else:
        print(f'reconnecting session {reconnect_session_key}')
        session_key = reconnect_session_key
        session_machine = SESSIONS[session_key]
        session_machine.add_connection(websocket)
        if session_machine.current_state == session_machine.inferring_stream:
            await session_machine.send_message(
                WSResponse(
                    is_error=False,
                    body=WSResponseStreamEstablished(
                        frame_url=f'/static/{session_machine.frame_name}')
                ).model_dump_json()
            )
    ###

    try:
        await websocket.send_message(WSResponse(
            body=WSResponseSessionConnected(session_key=session_key)
        ).model_dump_json())

        if session_machine.task_nursery is None:
            session_machine.task_nursery = trio.open_nursery()
            session_machine.set_channel(*trio.open_memory_channel(0))

        async with session_machine.task_nursery as n:
            assert n is not None

            # we don't need these workers for additional connections
            if reconnect_session_key is not None:
                # session_machine.task_nursery.start_soon(
                # parse_message, websocket, session_machine)
                await parse_message(websocket, session_machine)
            else:
                # async with trio.open_nursery() as n:

                # session_machine.task_nursery = loop

                # with trio.CancelScope() as cancel_scope:
                #     session_machine.cancel_scope = cancel_scope
                #     cancel_scope.shield = True
                n.start_soon(
                    message_worker, session_machine)
                n.start_soon(
                    spawn_watcher, session_machine)
                n.start_soon(parse_message,
                             websocket, session_machine)

                while session_machine.current_state != session_machine.session_terminated:
                    # logging.debug('[Process Daemon] Loop complete')

                    await trio.sleep(0)

    except* (ValidationError, JSONDecodeError, StreamNotValidError, cv2.error, StateInvariantError) as e:
        print(e)
        await websocket.send_message(WSResponse.from_error(e).model_dump_json())
        session_machine.cleanup()
    except* (ws.ConnectionClosed):
        print("conn closed")
    finally:
        pass
        # session_machine.cleanup()
        # del SESSIONS[session_key]


class StreamNotValidError(Exception):
    pass


class StateInvariantError(Exception):
    pass


class SessionNotFoundError(Exception):
    pass


async def message_worker(st: SessionState):
    # try:
    async for message in st.recv:
        if st.current_state == st.session_terminated:
            return

        logging.debug('got new message to send!')
        for sock in st.connections.copy():
            # with trio.move_on_after(2):
            try:
                await sock.send_message(message)
            except ws.ConnectionClosed:
                print(f'terminating connection {sock.CONNECTION_ID}')
                await st.remove_connection(sock)
        logging.debug('Done sending!')


async def parse_message(websocket: ws.WebSocketConnection, st: SessionState):
    # async for message in websocket:
    while not websocket.closed:
        logging.debug("[Parse Message Daemon] start loop")
        if st.current_state == st.session_terminated:
            return

        # with trio.move_on_after(0.5):
        try:
            message = await websocket.get_message()
        except ws.ConnectionClosed:
            print(f'terminating session {
                websocket=}, current conns {len(st.connections)}')
            await st.remove_connection(websocket)
            break

        if type(message) is bytes:
            img_format = filetype.image_match(message)
            if img_format is None:
                continue

            img_path = f'{st.session_token}-image.{img_format.EXTENSION}'

            with open(img_path, 'wb') as img_f:
                img_f.write(message)

            st.frame_recv()

            results = model(img_path, stream=True)

            for result in results:
                await st.send_message(
                    WSResponse(is_error=False, body=WSResponseInferResult(
                        image_shape=[result.orig_shape[1],
                                     result.orig_shape[0]],

                        data=result.tojson(),
                    )).model_dump_json()
                )
            os.remove(img_path)
            st.reset()
            continue

        if type(message) is not bytes and st.current_state == st.get_single_frame:
            await st.send_message(
                WSResponse(is_error=True, body=WSResponseError(error='Frame data should be binary.')).model_dump_json())

            st.reset()
            continue

        parsed: WSMessage = WSMessage.model_validate_json(message)
        if parsed.body.kind == 'INFER_FRAME' and st.current_state == st.ready:
            await st.send_message(
                (WSResponse(is_error=False, body=WSResponseSendBinary())).model_dump_json())
            st.infer_frame()

        if parsed.body.kind == 'INFER_STREAM':
            if st.current_state != st.ready:
                st.end_stream()
                raise StateInvariantError(
                    "State was not ready to accept a new stream!")

            target = parsed.body.stream_url
            st.stream_target = target
            st.infer_stream()

        if parsed.body.kind == 'STOP_STREAM':
            if st.current_state != st.inferring_stream:
                await st.send_message(WSResponse(
                    is_error=True,
                    body=WSResponseError(
                        error='State machine is not in the watching stream state!')
                ).model_dump_json())
                continue
            st.end_stream()

        if parsed.body.kind == 'REQUEST_NOTIFICATION':
            st.add_notification(parsed.body.extern_id,
                                parsed.body.conditions, email=parsed.body.email)


async def spawn_watcher(st: SessionState):
    while st.current_state != st.inferring_stream:
        if st.current_state == st.session_terminated:
            return
        logging.debug("[Stream Watcher] Watcher is still asleep...")
        await trio.sleep(0.25)

    target = st.stream_target

    if target.isnumeric():
        target = int(target)

    cap = cv2.VideoCapture(target)
    first = True
    st.capture = cap
    try:
        while cap.isOpened():
            if st.current_state != st.inferring_stream:
                cap.release()
                break

            ret, frame = cap.read()
            if not ret:
                raise StreamNotValidError("Could not read frame from stream!")
            h, w, l = frame.shape
            frame_res = cv2.resize(frame, (w//2, h//2))
            did_write = cv2.imwrite(
                st.frame_path,
                frame_res,
                params=[cv2.IMWRITE_JPEG_QUALITY, 65,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
            )
            if not did_write:
                raise StreamNotValidError("Could not write frame!")

            if first:
                await st.send_message(
                    WSResponse(
                        is_error=False,
                        body=WSResponseStreamEstablished(
                            frame_url=f'/static/{st.frame_name}')
                    ).model_dump_json()
                )
                first = False

            results: typing.Generator[Results] = model(
                frame, stream=True, verbose=False)

            for result in results:
                if len(result.boxes) == 0:
                    continue
                await st.send_message(
                    WSResponse(is_error=False, body=WSResponseInferResult(
                        data=result.tojson(),
                        image_shape=[result.orig_shape[1],
                                     result.orig_shape[0]]
                    )).model_dump_json()
                )
                await st.check_notifications(result)
            logging.debug("[Stream Watcher] Watcher is awake! ...")
            await trio.sleep(0.015)
    finally:
        if cap.isOpened():
            cap.release()
            # pass


async def handle(nursery: trio.Nursery, websocket: ws.WebSocketRequest):
    try:
        conn: ws.WebSocketConnection = await websocket.accept()
        print(f"new handle for websocket {conn.CONNECTION_ID}")
        message = await conn.get_message()
        message_parse = WSMessage.model_validate_json(message)
        if message_parse.body.kind not in [WSMessageStartSession.kind, WSMessageConnectSession.kind]:
            return
        # async with trio.open_nursery() as n:
        if type(message_parse.body) is WSMessageConnectSession:
            conn_partial = functools.partial(
                process,  reconnect_session_key=message_parse.body.session_key)
            await conn_partial(conn)
        else:
            await process(conn)

    except* (ValidationError, SessionNotFoundError) as e:
        await conn.send_message(WSResponse(is_error=True, body=WSResponseError(error=str(e.exceptions[0]))).model_dump_json())
        if type(e) is SessionNotFoundError:
            await process(conn)
    except* ws.ConnectionClosed:
        pass

SESSION_TIMEOUT_DISCONNECT = 5


async def session_monitor():
    while True:
        for key, state in SESSIONS.copy().items():
            if state.session_dead is not None:
                delta = abs(time.time() - state.session_dead)
                if delta < SESSION_TIMEOUT_DISCONNECT:
                    continue
                for conn in state.connections.copy():
                    if conn.closed:
                        state.connections.remove(conn)

                if len(state.connections) > 0:
                    # raise StateInvariantError(f"Session {
                    #   state.session_token} is marked for termination but there are still connections")
                    continue
                # run disconnects
                state.cleanup()
                state.terminate()
                del SESSIONS[key]
        logging.debug('[Session Daemon] Check complete')
        await trio.sleep(5)


async def main():
    async with trio.open_nursery() as n:

        serve = functools.partial(
            ws.serve_websocket, disconnect_timeout=0.5, ssl_context=None)
        handle_partial = functools.partial(handle, n)
        with trio.CancelScope() as cs:
            # cs.shield = True
            n.start_soon(serve,
                         handle_partial, "localhost", 8765)
        n.start_soon(session_monitor)

# print(WSMessage.model_json_schema(mode='serialization'))
# main_model_schema = WSMessage.model_json_schema(mode='serialization')  # (1)!
# print(json.dumps(main_model_schema, indent=2))  # (2)!

logger = logging.getLogger(__name__)

logging.basicConfig(filename='myapp.log', level=logging.INFO)
trio.run(main)
