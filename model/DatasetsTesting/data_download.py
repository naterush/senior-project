from img_utils import img_to_df
from scrape import API

def main():
    INPUT_IMG = "DatasetsTesting/Puerto_Rico_Biomass.img"
    OUTPUT_CSV = 'PR_Biomass_Coordinates_Dataset.csv'

    # The bounding box data can be gotten from the data download link, 
    # searching for: North_Bounding_Coordinate, etc.

    coord_biomass = img_to_df(
        INPUT_IMG,
        18.5542, # North
        17.7694, # South
        -65.13, # East Border
        -67.3228 # West Border,
    )

    coord_biomass.to_csv('PR_Biomass_Coordinates_Dataset.csv', index=False)

    # create an EE api
    api = API()

    # download sat images for all of these locations
    # NOTE: DON'T UNCOMMENT AS IT WILL DOWNLOAD 1 MILLION GBs
    """
    for index, row in coord_biomass.iterrows():
        api.download(row['latitude'], row['longitude'])
        # TODO: 
        # 1. have api.download return a path to a zip file that was downloaded
        # (and maybe some other meta-data like size, or covered region or something)
        # and then we down redownload data we already have
        # 2. write a "pre-process" function that takes this path and 
        #   a. opens the zip file, gets the specific image we are interested in
        #   b. reads in image, and pre-processes it how we see fit
        #   c. deletes the rest of the data and saves the final result
        # This will reduce the size of the data we are storing at any point in time
        # (which I think could get pretty large if we aren't careful...)
    """

    # TODO: feed this shit into a model???? idk
    



if __name__ == "__main__":
    main()


