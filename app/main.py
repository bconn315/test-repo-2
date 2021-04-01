from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app import ml

description = """
The Kickstarter-Success-Predictor application deploys data science 
to indicates the liklihood of success for proposed Kickstarter campaigns.

<img src="https://miro.medium.com/max/4638/1*nOdS52xlJh2n8T2Wu0UbKg.jpeg"
width="40%" />

"""

app = FastAPI(
    title='🏆 Kickstarter-Success-Predictor',
    description=description,
    docs_url='/',
)

app.include_router(ml.router, tags=['Model'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
