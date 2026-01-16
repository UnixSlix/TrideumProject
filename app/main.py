from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil
import os
import uuid

from promptEngineer import reportGeneration
from Rag_Module import process_and_index_new_files, clear_corpus_and_milvus

app = FastAPI()

app.mount("/generated_reports", StaticFiles(directory="generated_reports"), name="generated_reports")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.post("/uploadfile/")
async def create_upload_file(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    corpus_dir = "corpus"
    os.makedirs(corpus_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(corpus_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Run embedding and Milvus insert in the background
    background_tasks.add_task(process_and_index_new_files, corpus_dir)

    return {"message": f"{len(files)} file(s) uploaded and are being processed in the background."}

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    prompt: str = Form(...),
    template: str = Form(...),
    pages: int = Form(...)
):
    os.makedirs("generated_reports", exist_ok=True)
    report_id = uuid.uuid4().hex
    output_path = f"generated_reports/{report_id}"

    # pass the chosen template filename to reportGeneration
    reportGeneration(prompt, pages, output_path, template)

    return templates.TemplateResponse("home.html", {
        "request":     request,
        "response":    f"Report generated successfully for {pages} pages.",
        "report_path": f"{output_path}.doc",
        "pages":       pages,
        "template":    template
    })

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/clear")
def clear_corpus():
    clear_corpus_and_milvus()
    return JSONResponse(content={"message": "Corpus files deleted and Milvus collection cleared."})
