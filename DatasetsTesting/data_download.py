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
    """
    



if __name__ == "__main__":
    main()


