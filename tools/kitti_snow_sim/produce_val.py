import os

base_directory = "/home/hit/sdb1/Dataset/Ori_KITTI/sequences"
directory_paths = [os.path.join(base_directory, str(i).zfill(2)) for i in range(22)]

for directory_path in directory_paths:
    output_file_path = f"{directory_path}.txt"

    bin_path = directory_path + '/velodyne/'
    bin_files = [f for f in os.listdir(bin_path) if f.endswith('.bin')]

    file_names_without_extension = sorted([os.path.splitext(f)[0] for f in bin_files])

    with open(output_file_path, 'w') as output_file:
        for name in file_names_without_extension:
            output_file.write(name + "\n")

    print(f"All .bin file names from {directory_path} have been written to {output_file_path}.")
