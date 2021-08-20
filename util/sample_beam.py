import subprocess
import os
import sys
import time
import random
from tqdm import tqdm
import pp

file_dir = '/home/yanglei/GraphEditDistance/AIDS700nef/'

def command(cmd, timeout=60): 
    p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True) 
    t_beginning = time.time() 
    seconds_passed = 0 
    while True: 
        if p.poll() is not None: 
            break 
        seconds_passed = time.time() - t_beginning 
        if timeout and seconds_passed > timeout: 
            p.terminate() 
            return "-1\n".encode()
        time.sleep(1) 
    return p.stdout.read() 

def cal_ged(start,end,task,file_dir,timeout=60):
    result = ""
    for i in range(start, end):
        if random.random() > 0.05:
            continue
        beamsize = list(range(5,101,5))
        file1 = task[i][0]
        file2 = task[i][1]
        for beam in beamsize:
            if random.random() > 0.2:
                continue
            ged = 0
            for j in range(5):
                cmd_tmp = 'python src/ged.py '+file_dir+'train/'+file1+' '+file_dir+'test/'+file2+' BM '+ str(beam) +' 0'
                ged += int(command(cmd_tmp,timeout).decode())
            ged_norm = ged / 5
            result = result + file1 + '\t' + file2 + '\t' + str(beam) + '\t' + str(ged_norm) + '\n'
    return result

def load_task(file_dir):
    file_dir_train = file_dir+'train/'
    files_train = os.listdir(file_dir_train)
    file_dir_test = file_dir+'test/'
    files_test = os.listdir(file_dir_test)
    task = []
    for file1 in files_train:
        for file2 in files_test:
            if file1.split('.')[1] == 'gexf' and file2.split('.')[1] == 'gexf' and file1 != file2:
            # if file1.split('.')[1] == 'gxl' and file2.split('.')[1] == 'gxl':              
                task.append([file1,file2])
    print (len(task))
    return task

def load_task_trip(file_dir):
    file_dir_train = file_dir+'test/'
    files_train = os.listdir(file_dir_train)
    task = []
    for i in range(int(len(files_train)/3)):
        file1 = str(i)+'_1.gexf'
        file2 = str(i)+'_2.gexf'
        file3 = str(i)+'_3.gexf'
        if file1 in files_train and file2 in files_train and file3 in files_train:
            task.append([file1,file2])
            task.append([file1,file3])
    print (len(task))
    return task


def main():
    jobs = []
    task = load_task(file_dir)
    # task = load_task_trip(file_dir)
    start = 0
    timeout = 360
    end = len(task)
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 20
    step = end / ncpus
    job_server = pp.Server()
    job_server.set_ncpus(ncpus)
    for i in range(0, ncpus):
        ss = int(i * step)
        ee = int(ss + step)
        if ee > end:
            ee == end
        jobs.append(job_server.submit(cal_ged, (ss, ee, task, file_dir, timeout),(command,), modules=('subprocess','time','random',)))
    job_server.wait()
    results = ""
    for job in jobs:
        result = str(job())
        results += result

    with open(file_dir+'sample_beam.txt','w') as fout:
        fout.write(results)
    fout.close()
    print ("Successfully preprocess!")

if __name__ == "__main__":
    main()  
