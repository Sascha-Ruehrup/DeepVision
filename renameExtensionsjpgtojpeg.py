import glob, os

# https://stackoverflow.com/a/16736470/11670764

for filename in glob.iglob(os.path.join("images/0/", '*.jpg')):
    os.rename(filename, filename[:-4] + '.jpeg')
