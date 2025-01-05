import os
import subprocess
import threading
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from processor import process_video
from whisper_proccessor import transcribe_audio

app = FastAPI()

jobs = {}

@app.get("/jobs")
async def root():
    return {"message": "Hello World"}


@app.post("/job/new")
async def upload(request: Request):
    id = str(uuid.uuid4())

    form = await request.form()
    os.makedirs(f"jobs/{id}/")

    for field_name, file in form.items():
        try:
            with open(f'jobs/{id}/source.' + file.filename.split(".")[-1], 'wb') as f:
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)
                print("written")
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail='Something went wrong')
        finally:
            file.file.close()

    jobs[id] = {
        "id": id,
        "status": "Pending"
    }

    print("YEA")
    th = threading.Thread(target= __process, args=(id,))
    th.start()

    return {"id": id}

def __process(id):

    # 귀찮아 대충 그 나누는거
    jobs[id]["status"] = "Splitting"
    video_path = f"./jobs/{id}/source.mp4"
    audio_path = f"./jobs/{id}/source.wav"
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path, os.path.join(os.getcwd(), audio_path))
    subprocess.call(command, shell=True)

    # 그 STT하는거
    jobs[id]["status"] = "Transcribing"
    jobs[id]["transcript"] = transcribe_audio(os.path.join(os.getcwd(), audio_path))

    # 비디오 프로세싱
    jobs[id]["status"] = "Processing"
    jobs[id]["result"] = process_video(id, video_path, jobs[id]["transcript"])
    jobs[id]["status"] = "Done"

@app.get("/job/{id}")
def get_job(id: str):
    if not id in jobs:
        raise HTTPException(status_code=404)
    elif jobs[id]["status"] != "Done":
        raise HTTPException(status_code=425)
    else:
        return jobs[id]["result"]

@app.get("/job/{id}/stt")
def get_job_stt(id: str):
    if not id in jobs:
        raise HTTPException(status_code=404)
    elif "transcript" not in jobs[id]:
        raise HTTPException(status_code=425)
    else:
        return {"text": jobs[id]["transcript"]}

@app.get("/job/{id}/status")
def get_job_status(id: str):
    print(jobs)
    if id in jobs:
        return {"status": jobs[id]["status"]}
    else:
        raise HTTPException(status_code=404)

@app.get("/jobs/{id}/{filename}")
def get_job_file(id: str, filename: str):
    if not id in jobs:
        raise HTTPException(status_code=404)

    if filename.split(".")[-1] != "png":
        raise HTTPException(status_code=403)

    return FileResponse(path=os.getcwd() + f"/jobs/{id}/" + filename, media_type='application/octet-stream', filename=filename)