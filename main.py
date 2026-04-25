from rag import rag_pipeline
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
uploaded_file = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    global uploaded_file
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    uploaded_file = file.filename
    return {"filename": file.filename, "message": "PDF uploaded successfully"}

@app.post('/query')
def query(request: QueryRequest):
    question = request.question
    if uploaded_file == None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    try:
        answer = rag_pipeline(uploaded_file, question)
        return {"answer": answer}
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))