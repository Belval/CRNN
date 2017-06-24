import os
import sys

def main(argv):
    folder_name = argv[1]
    out_folder_name = argv[2]

    try:
        os.makedirs(out_folder_name)
    except:
        pass

    for top_folder in os.listdir(folder_name):
        for lower_folder in os.listdir(os.path.join(folder_name, top_folder)):
            for image in os.listdir(os.path.join(folder_name, top_folder, lower_folder)):
                os.rename(os.path.join(folder_name, top_folder, lower_folder, image), os.path.join(out_folder_name, image))

if __name__=='__main__':
    main(sys.argv)