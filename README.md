# How to run


### 1. Client
```sh
cd ws-print-client
npm i
npm run dev
# navigate to link given by vite
```

### 2.1 Infer server
```sh
echo "ONESIGNAL_KEY=<api_key>\nONESIGNAL_APP=<onesignal_app_id>" > .env
cd mini_print_stream_test
poetry install # or pip install -r requirements.txt
poetry shell
flask --app serve run &
python stream.py
```
Python 3.12.3 assumed. Use pyenv if you're on a different version. 
