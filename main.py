from rag import rag_pipeline
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
uploaded_files = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    filename: str

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get('/files')
def get_files():
    return {"files": uploaded_files}

@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    global uploaded_files
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDFs are supported")
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    uploaded_files.append(file.filename)
    return {"filename": file.filename, "message": "PDF uploaded successfully"}

@app.post('/query')
def query(request: QueryRequest):
    if len(uploaded_files) == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    try:
        answer = rag_pipeline(request.filename, request.question)
        return {"answer": answer}
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))