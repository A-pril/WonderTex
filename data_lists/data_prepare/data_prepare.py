import os
import csv
import json
import pickle


def read_json(file_path):
    with open(file_path) as f:
        json_list = json.load(f)
    
    return json_list

def read_csv_uid(file_path):
    uids = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile) 
        for row in csv_reader:
            uids.append(row[0])

    return uids

def list_to_json(list, file_path="./data_lists/lvis_uids_highquality_no3Dword.json"):
    json_data = json.dumps(list,ensure_ascii=False, indent=4)  

    # 将 json 数据写入文件
    with open(file_path, "w",encoding = 'utf-8') as file:
        file.write(json_data)

def get_no3dword(no3d_file):
    data_dir = "./data_lists/lvis_uids_filter_by_vertex.json"
    data_invalid_dir = "./data_lists/lvis_invalid_uids_nineviews.json"

    cap3d_dir = "./Cap3D/Cap3D_automated_Objaverse_highquality.csv"
    cap3d_no3d_dir = "./Cap3D/Cap3D_automated_Objaverse_no3Dword.csv"

    all_objects = read_json(data_dir)
    invalid_objects = read_json(data_invalid_dir)
    cap3d_objects = read_csv_uid(cap3d_dir) # 549922
    caped_no3d = read_csv_uid(cap3d_no3d_dir) # 661577

    # all valid objs
    objects = set(all_objects) - (set(invalid_objects) & set(all_objects))
    objects = list(objects)
    print(f"Download {len(objects)} objs") # 32371 valid objs in LVIS subset of objaverse

    # high quality objs in all valid objs
    highquality_objects = set(objects) & set(cap3d_objects)
    highquality_objects = list(highquality_objects)
    print(f"{len(highquality_objects)} high quality objs") # 26956 valid and high quality objs

    # no 3d word objs in all high quality objs
    no3d_objects = set(highquality_objects) & set(caped_no3d)
    no3d_objects = list(no3d_objects)
    print(f"{len(no3d_objects)} no3Dword high quality objs") # 25209 valid, high quality and no3dword objs

    list_to_json(no3d_objects, no3d_file)

    return no3d_objects

def main():
    no3d_file = "./data_lists/ours_uids_highquality_no3Dword.json"
    if not os.path.exists(no3d_file):
        no3d_objects = get_no3dword(no3d_file)
    else:
        no3d_objects = read_json(no3d_file)

    print(f"Final: {len(no3d_objects)} no3Dword high quality objs")

if __name__ == "__main__":
    main()