from itertools import islice
from multiprocessing import Process

if __name__ == "__main__":
    #write to file
    # output = open('aflw.list', 'w')
    #read faces rect from file
    faces_file = open('aflw.list_5', 'r')
    output = open('aflw.list.test', 'w')
    imageFaces = {}
    for line in faces_file.readlines():
        imagePath = line.split(' ')#[1].strip()
        if "non-face" in imagePath[0]:
            output.write(imagePath[0] + " " + str(0) + "\n")
        else:
            output.write(imagePath[0] + " " + str(1) + "\n")
    faces_file.close()
    output.close()