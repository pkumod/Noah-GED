import subprocess
import os
import sys
import time
from tqdm import tqdm
import pp
import re

file_dir = '/home/yanglei/GraphEditDistance/GREC/'
   
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

def cal_start(start,end,task,file_dir,timeout=60):
    result = ""
    for i in range(start, end):
        file1 = task[i][0]
        file2 = task[i][1]
        count = {}
        for j in range(100):
            if j >= 50:
                min_start_temp = max(count, key = count.get)
                if count[min_start_temp] + 100 - j < 50:
                    break
            cmd_tmp = 'python src/ged.py '+file_dir+'train/'+file1+' '+file_dir+'test/'+file2+' BM 10 0'
            min_start = re.findall(r'\d+', command(cmd_tmp,timeout).decode())        
            for k in min_start[:-1]:
                if k not in count:
                    count[k] = 1
                else:
                    count[k] += 1
        min_start_selected = max(count, key = count.get)
        if count[min_start_selected] > 50:
            result = result + file1 + '\t' + file2 + '\t' + str(min_start_selected) + '\n'
        else:
            continue
    return result

def load_task(file_dir):
    file_dir_train = file_dir+'train/'
    files_train = os.listdir(file_dir_train)
    file_dir_test = file_dir+'test/'
    files_test = os.listdir(file_dir_test)
    task = []
    for file1 in files_train:
        for file2 in files_test:
            # if file1.split('.')[1] == 'gexf' and file2.split('.')[1] == 'gexf' and file1 != file2:
            if file1.split('.')[1] == 'gxl' and file2.split('.')[1] == 'gxl' and file1 != file2:              
                task.append([file1,file2])
    print (len(task))
    return task

def main():
    jobs = []
    task = load_task(file_dir)
    start = 0
    timeout = 360
    end = len(task)
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 30
    step = end / ncpus
    job_server = pp.Server()
    job_server.set_ncpus(ncpus)
    for i in range(0, ncpus):
        ss = int(i * step)
        ee = int(ss + step)
        if ee > end:
            ee == end
        jobs.append(job_server.submit(cal_start, (ss, ee, task, file_dir, timeout),(command,), modules=('subprocess','time','re')))
    job_server.wait()
    results = ""
    for job in jobs:
        result = str(job())
        results += result

    with open(file_dir+'start.txt','w') as fout:
        fout.write(results)
    fout.close()
    print ("Successfully preprocess!")

if __name__ == "__main__":
    main()  
