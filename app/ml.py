"""Machine learning functions."""

import logging

from joblib import load
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

# uvicorn app.main:app --reload

classifier = load("app/classifier.joblib")


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    category: str = Field(..., example="Food")
    main_category: str = Field(..., example="Drink")
    backers: int = Field(..., example=25)
    usd_goal_real: float = Field(..., example=5000.00)
    ks_length: int = Field(..., example=60)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    # @validator('x3')
    # def x3_must_be_positive(cls, value):
    #     """Validate that x3 is a positive number."""
    #     assert value > 0, f'x3 == {value}, must be > 0'
    #     return value


@router.post("/predict")
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🎱.

    ### Request Body
    - `category`: Select a category ['3D Printing', 'Academic', 'Accessories', 'Action', 'Animals', 'Animation', 'Anthologies', 
        'Apparel', 'Apps', 'Architecture', 'Art', 'Art Books', 'Audio', 'Bacon', 
        'Blues', 'Calendars', 'Camera Equipment', 'Candles', 'Ceramics', 'Childrens Books', 'Childrenswear', 
        'Chiptune', 'Civic Design', 'Classical Music', 'Comedy', 'Comic Books', 'Comics', 'Community Gardens', 
        'Conceptual Art', 'Cookbooks', 'Country & Folk', 'Couture', 'Crafts', 'Crochet', 'Dance', 
        'Design', 'Digital Art', 'DIY', 'DIY Electronics', 'Documentary', 'Drama', 'Drinks', 'Electronic Music', 
        'Embroidery', 'Events', 'Experimental', 'Fabrication Tools', 'Faith', 'Family', 'Fantasy', 
        'Farmers Markets', 'Farms', 'Fashion', 'Festivals', 'Fiction', 'Film & Video', 'Fine Art', 
        'Flight', 'Food', 'Food Trucks', 'Footwear', 'Gadgets', 'Games', 'Gaming Hardware', 
        'Glass', 'Graphic Design', 'Graphic Novels', 'Hardware', 'Hip-Hop', 'Horror', 'Illustration', 
        'Immersive', 'Indie Rock', 'Installations', 'Interactive Design', 'Jazz', 'Jewelry', 'Journalism', 
        'Kids', 'Knitting', 'Latin', 'Letterpress', 'Literary Journals', 'Literary Spaces', 'Live Games', 
        'Makerspaces', 'Metal', 'Mixed Media', 'Mobile Games', 'Movie Theaters', 'Music', 'Music Videos', 
        'Musical', 'Narrative Film', 'Nature', 'Nonfiction', 'Painting', 'People', 'Performance Art', 
        'Performances', 'Periodicals', 'Pet Fashion', 'Photo', 'Photobooks', 'Photography', 'Places', 
        'Playing Cards', 'Plays', 'Poetry', 'Pop', 'Pottery', 'Print', 'Printing', 
        'Product Design', 'Public Art', 'Publishing', 'Punk', 'Puzzles', 'Quilts', 'R&B', 
        'Radio & Podcasts', 'Ready-to-wear', 'Residencies', 'Restaurants', 'Robots', 'Rock', 'Romance', 
        'Science Fiction', 'Sculpture', 'Shorts', 'Small Batch', 'Software', 'Sound', 'Space Exploration', 
        'Spaces', 'Stationery', 'Tabletop Games', 'Taxidermy', 'Technology', 'Television', 'Textiles', 
        'Theater', 'Thrillers', 'Translations', 'Typography', 'Vegan', 'Video', 'Video Art', 
        'Video Games', 'Wearables', 'Weaving', 'Web', 'Webcomics', 'Webseries', 'Woodworking', 
        'Workshops', 'World Music', 'Young Adult', 'Zines']

    - `main_category`: Select a main category ['Art', 'Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film & Video', 
        'Food', 'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater']

    - `backers`: Estimate the number of expected backers for your Kickstarter
        project

    - `usd_goal_real`: Enter prospective Kickstarter campaign's fundraising
        goal in U.S. dollars

    - `ks_length`: Enter the duration of proposed Kickstarter fundraising
        campaign in days

    ### Response
    - `prediction`: success, failed
    - `predict_proba`: float between 0.0 and 1.0,
    representing the predicted class's probability

    """

    X_new = item.to_df()
    choice = classifier.predict(X_new)
    probability = classifier.predict_proba(X_new)
    return choice[0], f"{probability[0][1]*100:.2f}% probability"
