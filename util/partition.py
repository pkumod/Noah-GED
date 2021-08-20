import xml.etree.ElementTree as ET
import os
import shutil
import sys

def loadCXL(filename):
    files = {}
    tree = ET.parse(filename)
    root = tree.getroot()
    for file in root[0].iter('print'):
        files[file.attrib['file']] = file.attrib['class']
    return files

def main():
    if len(sys.argv) != 3:
        print ("Usage: python partition.py DATA_DIR PARTITION_TYPE")
        exit(0)
    dir = sys.argv[1]
    type = sys.argv[2]
    if not dir.endswith('/'):
        dir = dir + '/'
    target_dir = dir + type + '/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)
    files = loadCXL(dir + type + '.cxl')
    total = os.listdir(dir)
    for (k,v) in files.items():
        if k in total:
            shutil.copyfile(os.path.join(dir,k), os.path.join(target_dir,k))

if __name__ == "__main__":
    main()