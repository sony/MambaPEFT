import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import personal_lib


from mmengine.runner import find_latest_checkpoint

def select_main_file():
    # this is the top level parser
    print(sys.argv)
    task_ind = 1
    if sys.argv[1].startswith("--local_rank") or sys.argv[1].startswith("--local-rank"): #distributed trainingでここに--local_rank変数が自動挿入される場合あり
        task_ind = 2
    
    # last_checkpointというファイルに直近のcheckpoint名が書かれており、そのファイルに置き換える
    for i in range(len(sys.argv)):
        if sys.argv[i].endswith("last_checkpoint"):
            filename=sys.argv[i]
            if os.path.exists(filename):
                latest_filename = find_latest_checkpoint(filename.replace("last_checkpoint", ""))
            else: #for mmlab 1.x compatibility
                latest_filename = filename.replace("last_checkpoint", "latest.pth")
            if "/work_dirs/" in latest_filename: # absolute path to relative path for different root dir
                latest_filename = "work_dirs/" + "/work_dirs/".join(latest_filename.split("/work_dirs/")[1:])
                if not os.path.exists(latest_filename): # gaia_tools
                    latest_filename = "../" + latest_filename
            print(f"latest checkpoint {latest_filename} is found")
            sys.argv[i] = latest_filename

    if sys.argv[task_ind] == "mmpretrain":
        from mmpretrain_tools.test import main
        sys.argv.pop(task_ind)
        print(sys.argv)
        main()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    select_main_file()