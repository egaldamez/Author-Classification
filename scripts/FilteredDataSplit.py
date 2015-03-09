import os

# ---------------
# NOTE: You should delete the files out of the chunk directory before running
# ---------------

# Set size of file chunks in kB
CHUNK_SIZE_KB = 5

# Iterates through the chunks of a file
def chunk(filename):
    with open(filename) as f:
        for part in iter(lambda: f.read(1024 * CHUNK_SIZE_KB), ''):
            yield(part)

if __name__ == "__main__":
    data_directory = "..\\filtereddata"
    chunk_directory = "..\\chunkdata"

    # Iterate through all files in given directory and create new files in
    # other given directory for each chunk (Might not work with subdirectories as is)
    for dirname, subdirnames, filenames in os.walk(data_directory):

        for file in filenames:
            i = 1
            for part in chunk(os.path.join(dirname, file)):
                with open(os.path.join(chunk_directory, file + "-part{}.txt".format(i)), 'w') as new_f:
                    new_f.write(part)
                i += 1
