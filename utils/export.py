
from time import asctime

def export_submission(data,model):
    f = open("submissions/submission-" + asctime()+ ".csv", "w+")
    f.write("playlist_id,track_ids\n")
    for ind, playlist_id in enumerate(data['playlist_id']):
        f.write(str(playlist_id) + ',' + model.recommend(playlist_id, at=10) + '\n');
    f.close()

