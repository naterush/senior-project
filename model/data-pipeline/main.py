import sys
import pickle
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
    print("Downloading the scenes...")
    old_scene = api.download(
        lat, 
        lng, 
        num_scenes=1,
        start_date="2016-1-1",
        end_data="2017-1-1"
    )[0]

    new_scene = api.download(
        lat, 
        lng, 
        num_scenes=1,
        start_date="2018-1-1",
        end_data="2019-1-1"
    )[0]


    old_labels = old_scene.label()
    old_preds = 
    labels = []
    # label all the scenes we got
    for scene in downloaded_scenes:
        label = scene.label()
        labels.append(label)

    # run these through the model
    fn = 'logreg_model_41620.sav'
    loaded_model = pickle.load(open(fn, 'rb'))
    preds = loaded_model.predict(labels[0][:, :3])
    num_correct = len(preds[preds==labels[0][:, 3]])
    acc = num_correct/len(preds)
    print (str(num_correct) + " out of " + str(len(preds)))
    print("Accuracy: ", acc)

    # then, print them to the screen




if __name__ == "__main__":
    main()