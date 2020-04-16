import sys
from scrape import LandsatAPI


def main():
    if (len(sys.argv)) != 3:
        print("Usage: python3 main.py <lat> <long>")
        return

    lat = float(sys.argv[1])
    lng = float(sys.argv[2])

    # make the api
    api = LandsatAPI()

    # download the image
    downloaded_scenes = api.download(lat, lng, num_scenes=1)

    # extract them all
    for scene in downloaded_scenes:
        scene.write_img()

    # run these through the model

    # then, print them to the screen

    


if __name__ == "__main__":
    main()