import requests
import pandas as pd


def download_images(img_urls, output_folder):
    total = 0

    successful_downloads = {}

    p = ""

    for url, class_name in img_urls.items():
        # loop the URLs
        try:
            # try to download the image
            r = requests.get(url, timeout=5)
            # save the image to disk
            p = f"./{output_folder}/{class_name}/{str(total).zfill(8)}.jpg"
            f = open(p, "wb")
            f.write(r.content)
            f.close()
            # update the counter
            print("[INFO] downloaded: {}".format(p))

            successful_downloads[p] = {"index_no":total,  "label": class_name}
            total += 1
        # handle if any exceptions are thrown during the download process
        except Exception as e:
            print("[INFO] error downloading {}...skipping".format(e))
        
    # At end save csv file with info on all images
    df = pd.DataFrame.from_dict(successful_downloads, orient='index', columns=["index_no", "label"])
    df.to_csv(f"./{output_folder}/not_melon_all_imgs.csv", index=True)

    # Return status message
    if len(successful_downloads) > 1:
        return "Success!"
    else: 
        return "Failed"




if __name__ == "__main__":
    files_dir = "./melon_urls/"
    # files_name = ["object", "apple", "orange", "people"]
    files_name = ["honeydew", "cantaloupe", "cantaloupe2", "melon", "watermelon", "watermelon2", "watermelon3"]
    
    img_urls = {}

    import fileinput

    for fn in files_name:
        with fileinput.input(files=f"{files_dir}{fn}.txt") as f:
            for line in f:
                img_urls[line.strip("\n")] = fn
            
    # Dictionary format e.g. : img_urls[url] = "honeydew"

    download_images(img_urls, "not_melon_imgs")
    


            

    