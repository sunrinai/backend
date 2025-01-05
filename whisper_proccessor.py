import whisper

def transcribe_audio(file_path):
    """
    Whisper를 사용하여 텍스트 변환 (전체 텍스트 출력)
    """
    model = whisper.load_model("turbo")
    result = model.transcribe(file_path)
    return result["text"]